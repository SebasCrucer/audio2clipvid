import os
import whisperx
import torch
import moviepy as mp
from typing import Callable, List, Tuple
from clipvid import ClipVidSearcher

def generate_video(
        audio_path: str,
        llm_function: Callable[[str], str],
        embeddings_folder: str,
        videos_folder: str,
        output_video_path: str,
        device: str = None,
        max_segment_words: int = 40
):
    """
    Convierte un audio en un video usando WhisperX para extraer timestamps por palabra,
    una función (lambda) LLM para fragmentar el texto en ideas completas y generar descripciones,
    y ClipVidSearcher para buscar y componer clips.

    - Si el LLM devuelve más palabras de las que realmente existen en la transcripción,
      truncamos o removemos los fragmentos sobrantes (para evitar IndexError).
    - Extendemos cada fragmento hasta el inicio del siguiente fragmento (para no dejar huecos).
    - Añadimos un fragmento FILLER para cubrir el silencio final, si lo hay.
    """

    print("[generate_video] INICIO")
    print(f"[generate_video] Parámetros:\n  audio_path={audio_path}\n  embeddings_folder={embeddings_folder}\n  videos_folder={videos_folder}\n  output_video_path={output_video_path}\n  device={device}\n  max_segment_words={max_segment_words}")

    # 1. Determinar dispositivo (CPU o GPU)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[generate_video] Usando dispositivo: {device}")

    # 2. Transcribir audio con WhisperX
    print("[generate_video] Transcribiendo audio...")
    transcribed_text, word_timestamps = transcribe_audio_with_whisperx(audio_path, device=device)
    print(f"[generate_video] Texto transcrito (primeros 200 caracteres): {transcribed_text[:200]}...")
    print(f"[generate_video] Cantidad de palabras detectadas por WhisperX: {len(word_timestamps)}")

    # 3. Dividir el texto en fragmentos con el LLM
    print("[generate_video] Dividiendo texto en fragmentos con el LLM...")
    splitted_fragments = split_script_into_fragments(
        transcribed_text,
        llm_function,
        max_words=max_segment_words
    )
    print("[generate_video] Fragments recibidos del LLM:")
    for i, frag in enumerate(splitted_fragments):
        print(f"  Fragmento {i}: {frag[:100]}...")

    # 4. Ajustar fragmentos para NO exceder las palabras totales de whisperx
    print("[generate_video] Ajustando fragmentos para que no excedan palabras de WhisperX...")
    splitted_fragments = adjust_fragments_to_whisperx(
        splitted_fragments,
        word_timestamps
    )
    print("[generate_video] Fragments luego del ajuste:")
    for i, frag in enumerate(splitted_fragments):
        print(f"  Fragmento {i}: {frag[:100]}...")

    # 5. Cargar el audio completo y medir duración
    print("[generate_video] Cargando audio para medir duración...")
    audio_clip = mp.AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    print(f"[generate_video] Duración del audio: {audio_duration} segundos")

    # 6. Calcular rangos de tiempo con extensión hasta el siguiente fragmento
    print("[generate_video] Calculando rangos de tiempo para cada fragmento...")
    fragment_time_ranges = compute_fragment_time_ranges(
        word_timestamps,
        splitted_fragments,
        audio_duration
    )
    print("[generate_video] Rangos de tiempo calculados:")
    for i, (start, end) in enumerate(fragment_time_ranges):
        print(f"  Rango {i}: start={start}, end={end}")

    # 7. Generar descripción corta para cada fragmento (incluyendo FILLER)
    print("[generate_video] Generando descripción corta con LLM para cada fragmento...")
    short_descriptions = []
    for i, fragment in enumerate(splitted_fragments):
        if fragment == "FILLER":
            desc = "Silence or filler"
            print(f"  [Frag {i}] Es FILLER => desc='{desc}'")
        else:
            prompt_desc = f"Genera una breve descripción en inglés de hasta 5 palabras clave para: {fragment}"
            desc = llm_function(prompt_desc)
            print(f"  [Frag {i}] Descripción generada: {desc}")
        short_descriptions.append(desc.strip())

    # 8. Crear el buscador de videos con ClipVidSearcher
    print("[generate_video] Creando instancia de ClipVidSearcher...")
    searcher = ClipVidSearcher(
        embeddings_folder=embeddings_folder,
        videos_folder=videos_folder,
        device=device
    )

    # 9. Para cada fragmento y descripción, generar un clip que cubra su duración
    print("[generate_video] Generando clips para cada fragmento...")
    segment_clips = []
    used_videos = set()

    for i, ((frag, desc), (start, end)) in enumerate(zip(zip(splitted_fragments, short_descriptions), fragment_time_ranges)):
        required_duration = end - start
        print(f"  [Segment {i}] Fragment='{frag[:50]}...', Desc='{desc}', Duración requerida={required_duration}")
        clip_segment = compose_segment_with_searcher(
            searcher=searcher,
            description=desc,
            duration=required_duration,
            used_videos=used_videos,
        )
        segment_clips.append(clip_segment)

    # 10. Concatenar los clips de todos los fragmentos
    print("[generate_video] Concatenando todos los fragmentos de video...")
    final_video_clip = mp.concatenate_videoclips(segment_clips, method="compose")

    # 11. Asignar el audio original (no lo recortamos, pues cubrimos cualquier silencio con FILLER o con extensión)
    print("[generate_video] Asignando audio original al video concatenado...")
    final_video_clip.audio = audio_clip

    # 12. Exportar el video
    print("[generate_video] Exportando video final...")
    final_video_clip.write_videofile(
        output_video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )

    # 13. Cerrar recursos
    print("[generate_video] Cerrando recursos (video_clip y audio_clip)...")
    final_video_clip.close()
    audio_clip.close()
    for sc in segment_clips:
        sc.close()

    print(f"[generate_video] Video generado en: {output_video_path}")
    print("[generate_video] FIN")


def transcribe_audio_with_whisperx(audio_path: str, device: str = "cpu"):
    """
    Transcribe el audio usando WhisperX y retorna:
      - El texto completo transcrito
      - Una lista con (palabra, start_time, end_time)
    """
    print("[transcribe_audio_with_whisperx] INICIO")
    print(f"[transcribe_audio_with_whisperx] Parámetros:\n  audio_path={audio_path}\n  device={device}")

    model = whisperx.load_model("small", device=device)
    print("[transcribe_audio_with_whisperx] Modelo WhisperX cargado (small).")

    audio = whisperx.load_audio(audio_path)
    print("[transcribe_audio_with_whisperx] Audio cargado para transcripción.")

    # Transcribir
    print("[transcribe_audio_with_whisperx] Iniciando transcripción...")
    result = model.transcribe(audio)
    print("[transcribe_audio_with_whisperx] Transcripción base finalizada.")

    # Alinear las palabras con timestamps
    print("[transcribe_audio_with_whisperx] Cargando modelo de alineación...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    print("[transcribe_audio_with_whisperx] Modelo de alineación cargado.")

    print("[transcribe_audio_with_whisperx] Alineando segmentos para obtener timestamps de palabras...")
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio_path, device,
        return_char_alignments=False
    )

    # Reconstruir texto y extraer timestamps
    print("[transcribe_audio_with_whisperx] Reconstruyendo texto y extrayendo timestamps de palabras...")
    word_timestamps = []
    all_text = []
    for seg in result_aligned["segments"]:
        for w in seg["words"]:
            word_timestamps.append((w["word"], w["start"], w["end"]))
            all_text.append(w["word"])

    transcript_text = " ".join(all_text)
    print("[transcribe_audio_with_whisperx] INICIO - Texto transcrito completo:")
    print(transcript_text)
    print("[transcribe_audio_with_whisperx] FIN - Texto transcrito completo")
    print(f"[transcribe_audio_with_whisperx] Cantidad de palabras detectadas: {len(word_timestamps)}")
    print("[transcribe_audio_with_whisperx] FIN")

    return transcript_text, word_timestamps


def split_script_into_fragments(full_text: str, llm_function: Callable[[str], str], max_words: int = 40):
    """
    Usa la función llm_function para generar la división del texto en fragmentos
    de un cierto tamaño (en palabras), intentando agrupar ideas completas.

    NOTA: El LLM podría devolver más (o menos) palabras que las reales,
    así que luego debemos ajustar.
    """
    print("[split_script_into_fragments] INICIO")
    print(f"[split_script_into_fragments] Parámetros:\n  max_words={max_words}")
    print("[split_script_into_fragments] Preparando prompt para LLM...")

    prompt = (
        f"Por favor, divide el siguiente texto en fragmentos, "
        f"manteniendo las ideas completas en cada fragmento. "
        f"Evita cortar frases que formen una unidad de sentido. "
        f"Cada fragmento debe estar compuesto por alrededor de {max_words} "
        f"palabras como referencia, pero prioriza no interrumpir la "
        f"coherencia del contenido. Devuelve cada fragmento en una nueva línea. "
        f"Texto: {full_text}"
    )

    print("[split_script_into_fragments] Llamando LLM function...")
    response = llm_function(prompt)

    print("[split_script_into_fragments] Respuesta completa del LLM:")
    print(response)

    fragments = [line.strip() for line in response.split("\n") if line.strip()]
    print(f"[split_script_into_fragments] Total de fragmentos recibidos: {len(fragments)}")
    print("[split_script_into_fragments] FIN")
    return fragments


def adjust_fragments_to_whisperx(fragments: List[str], word_timestamps: List[Tuple[str, float, float]]):
    """
    Ajusta la lista de fragmentos generados por el LLM para que NO excedan
    la cantidad total de palabras que WhisperX detectó.

    - Si sobran palabras, se recorta el último fragmento (o se elimina el fragmento completo)
      para evitar IndexError. Esto podría truncar la idea final del LLM, pero se hace
      para garantizar la validez de timestamps.

    - Si hay menos palabras de las que detectó WhisperX, NO hacemos nada especial,
      pues se cubre con FILLER o extensión final en compute_fragment_time_ranges.
    """
    print("[adjust_fragments_to_whisperx] INICIO")
    total_whisperx_words = len(word_timestamps)
    print(f"[adjust_fragments_to_whisperx] Cantidad total de palabras en WhisperX: {total_whisperx_words}")

    # Contar cuántas palabras totales devolvió el LLM
    splitted_fragment_tokens = [frag.split() for frag in fragments]
    all_llm_tokens = [token for tokens in splitted_fragment_tokens for token in tokens]
    total_llm_words = len(all_llm_tokens)
    print(f"[adjust_fragments_to_whisperx] Cantidad total de palabras en fragments LLM: {total_llm_words}")

    # Si el LLM no excede, está todo bien
    if total_llm_words <= total_whisperx_words:
        print("[adjust_fragments_to_whisperx] No excede, no se recorta nada.")
        print("[adjust_fragments_to_whisperx] FIN")
        return fragments

    # Caso: LLM generó más palabras de las que existen => recortar
    difference = total_llm_words - total_whisperx_words
    print(f"[adjust_fragments_to_whisperx] Excedente: {difference} palabras")

    # Trabajamos desde el último fragmento hacia atrás
    i = len(fragments) - 1
    while difference > 0 and i >= 0:
        tokens = fragments[i].split()
        if len(tokens) <= difference:
            # Eliminar fragmento completo (pues excede)
            print(f"[adjust_fragments_to_whisperx] Eliminando fragmento completo en índice {i} con {len(tokens)} tokens.")
            fragments.pop(i)
            difference -= len(tokens)
            i -= 1
        else:
            # Quitar 'difference' tokens del final de ese fragmento
            print(f"[adjust_fragments_to_whisperx] Recortando {difference} tokens del fragmento índice {i}.")
            kept_tokens = tokens[:-difference]
            fragments[i] = " ".join(kept_tokens)
            difference = 0  # ya no necesitamos quitar más

    print("[adjust_fragments_to_whisperx] FIN")
    return fragments


def compute_fragment_time_ranges(
        word_timestamps: List[Tuple[str, float, float]],
        fragments: List[str],
        audio_duration: float
):
    """
    Dado el array (palabra, start, end) y la lista de fragmentos (ajustada para no exceder),
    asigna rangos de tiempo a cada fragmento (start_time, end_time), **extendiendo** cada uno
    hasta el inicio del siguiente para evitar huecos.

    Además, si sobra tiempo después del último fragmento, añadimos
    un fragmento "FILLER" para cubrir [fin_último_fragmento, fin_audio].
    """
    print("[compute_fragment_time_ranges] INICIO")
    print(f"[compute_fragment_time_ranges] Cantidad de word_timestamps: {len(word_timestamps)}")
    print(f"[compute_fragment_time_ranges] Cantidad de fragments: {len(fragments)}")
    print(f"[compute_fragment_time_ranges] audio_duration={audio_duration}")

    fragment_time_ranges = []
    index_palabra = 0

    # Recorremos los fragments, pero debemos ver si hay un siguiente fragment
    # para extender el end_time. Guardamos primero el "start_time" del fragment actual
    # y luego calculamos el "start_time" del siguiente fragment para cerrar la brecha.
    for i, fragment in enumerate(fragments):
        if fragment == "FILLER":
            # Ignoramos aquí, se maneja al final si hace falta
            continue

        word_count = len(fragment.split())
        if word_count == 0:
            print(f"[compute_fragment_time_ranges] Fragmento {i} vacío, se omite.")
            continue

        # 1) Start_time de este fragmento => start de su primera palabra
        start_time = word_timestamps[index_palabra][1]

        # 2) Miramos si hay un siguiente fragmento 'hablado' (no FILLER, no vacío).
        #    Si lo hay, el end_time de este fragmento será el start_time de la primera palabra
        #    de ese fragmento. Así cubrimos los silencios intermedios.
        #    Si NO lo hay, el end_time será la última palabra de ESTE fragmento.
        next_frag_index = find_next_nonfiller_fragment(fragments, i+1)
        if next_frag_index is not None:
            # El end_time de este fragmento es el start_time de la primera palabra
            # de ese "siguiente fragmento".
            next_word_count = len(fragments[next_frag_index].split())

            # Este fragmento abarca 'word_count' palabras a partir de index_palabra,
            # el siguiente empieza en index_palabra + word_count
            # (si no nos salimos de la lista de word_timestamps).
            next_fragment_first_word_idx = index_palabra + word_count
            if next_fragment_first_word_idx < len(word_timestamps):
                end_time = word_timestamps[next_fragment_first_word_idx][1]
            else:
                # En caso de que no haya más palabras, lo llevamos hasta su última palabra.
                end_time = word_timestamps[index_palabra + word_count - 1][2]
        else:
            # No hay siguiente fragmento 'hablado'. Tomamos la última palabra de ESTE.
            end_time = word_timestamps[index_palabra + word_count - 1][2]

        fragment_time_ranges.append((start_time, end_time))

        # Aumentamos el índice para el siguiente
        index_palabra += word_count

    # Comprobamos hasta dónde llegamos con el último fragmento
    last_range_end = fragment_time_ranges[-1][1] if fragment_time_ranges else 0.0

    # Si la duración del audio es mayor que el fin del último fragmento, añadimos FILLER
    if audio_duration > last_range_end:
        # Si no existe FILLER al final, lo añadimos
        if not fragments or fragments[-1] != "FILLER":
            print("[compute_fragment_time_ranges] Añadiendo FILLER al final de la lista de fragments.")
            fragments.append("FILLER")
        fragment_time_ranges.append((last_range_end, audio_duration))
        print(f"[compute_fragment_time_ranges] Rango FILLER: start={last_range_end}, end={audio_duration}")

    print("[compute_fragment_time_ranges] FIN")
    return fragment_time_ranges


def find_next_nonfiller_fragment(fragments: List[str], start_idx: int) -> int:
    """
    Dado un índice de inicio, busca el siguiente fragmento
    que no sea "FILLER" ni esté vacío. Retorna el índice o None si no hay.
    """
    for i in range(start_idx, len(fragments)):
        if fragments[i] != "FILLER" and len(fragments[i].split()) > 0:
            return i
    return None


def ajustar_clip_vertical(clip: mp.VideoFileClip, target_w=1080, target_h=1920):
    """
    Ajusta un clip al formato vertical 1080x1920 (9:16).
    """
    print("[ajustar_clip_vertical] INICIO")
    cw, ch = clip.size
    print(f"[ajustar_clip_vertical] Tamaño original clip: width={cw}, height={ch}")
    target_ratio = target_w / target_h
    clip_ratio = cw / ch

    if clip_ratio > target_ratio:
        # El clip es más ancho que 9:16; escalamos por altura y recortamos a los lados
        scaled_clip = clip.resized(height=target_h)
        new_w, new_h = scaled_clip.size
        x_center = new_w / 2
        x1 = x_center - (target_w / 2)
        x2 = x_center + (target_w / 2)
        final_clip = scaled_clip.cropped(x1=x1, y1=0, x2=x2, y2=new_h)
        print("[ajustar_clip_vertical] Clip recortado a los lados para formato 9:16.")
    else:
        # El clip es más alto o igual al ratio; escalamos por ancho y recortamos
        scaled_clip = clip.resized(width=target_w)
        new_w, new_h = scaled_clip.size
        print(f"[ajustar_clip_vertical] Después de escalar por ancho: {new_w}x{new_h}")
        if new_h > target_h:
            y_center = new_h / 2
            y1 = y_center - (target_h / 2)
            y2 = y_center + (target_h / 2)
            final_clip = scaled_clip.cropped(x1=0, y1=y1, x2=new_w, y2=y2)
            print("[ajustar_clip_vertical] Clip recortado verticalmente para formato 9:16.")
        else:
            final_clip = scaled_clip.resized((target_w, target_h))
            print("[ajustar_clip_vertical] Clip escalado a 9:16 directamente, sin recorte adicional.")

    print("[ajustar_clip_vertical] FIN")
    return final_clip


def compose_segment_with_searcher(
        searcher,
        description: str,
        duration: float,
        used_videos=None,
        target_w=1080,
        target_h=1920
) -> mp.VideoClip:
    """
    Usa el buscador ClipVidSearcher para encontrar uno o varios videos
    que coincidan con `description`. Se concatenan hasta cubrir `duration`.
    - Si used_videos existe, se evita reutilizar un video en distintos segmentos.
    - Ajusta cada clip a 9:16 con ajustar_clip_vertical().
    - Recorta el último clip si excede la duración necesaria.
    - Si no se encuentra nada, retorna un ColorClip de relleno.
    """
    print("[compose_segment_with_searcher] INICIO")
    print(f"[compose_segment_with_searcher] Parámetros:\n  description={description}\n  duration={duration}\n  used_videos={used_videos}\n  target_w={target_w}\n  target_h={target_h}")

    if used_videos is None:
        used_videos = set()

    # Buscar top 10 videos
    print("[compose_segment_with_searcher] Buscando videos con descripción...")
    results = searcher.search(description, n=10)  # [(sim, idx, path), ...]
    print(f"[compose_segment_with_searcher] Resultados encontrados: {len(results)}")

    used_clips = []
    accumulated_duration = 0.0
    idx = 0

    # Ir concatenando hasta cubrir la duración
    while accumulated_duration < duration and idx < len(results):
        _, _, video_path = results[idx]
        idx += 1

        # Evitar duplicados
        if video_path in used_videos:
            print(f"[compose_segment_with_searcher] Video repetido, ignorando: {video_path}")
            continue
        used_videos.add(video_path)

        if not os.path.exists(video_path):
            print(f"[compose_segment_with_searcher] Archivo no existe: {video_path}")
            continue

        print(f"[compose_segment_with_searcher] Usando video: {video_path}")
        clip = mp.VideoFileClip(video_path)
        clip = ajustar_clip_vertical(clip, target_w, target_h)

        clip_needed = duration - accumulated_duration
        if clip.duration > clip_needed:
            print(f"[compose_segment_with_searcher] Clip excede lo requerido. Recortando a {clip_needed} seg.")
            clip = clip.subclipped(0, clip_needed)

        used_clips.append(clip)
        accumulated_duration += clip.duration
        print(f"[compose_segment_with_searcher] Duración acumulada en subclips: {accumulated_duration}")

    # Si no se encontró nada o no se cubrió la duración
    if not used_clips:
        print("[compose_segment_with_searcher] No se encontraron videos adecuados o ninguno usable. Retornando ColorClip.")
        filler_clip = mp.ColorClip(size=(target_w, target_h), color=(0, 0, 0), duration=duration)
        print("[compose_segment_with_searcher] FIN")
        return filler_clip

    # Concatenar todos los subclips
    if len(used_clips) == 1:
        final_clip = used_clips[0]
        print("[compose_segment_with_searcher] Sólo un clip en used_clips. No concateno múltiples.")
    else:
        print("[compose_segment_with_searcher] Concatenando múltiples subclips...")
        final_clip = mp.concatenate_videoclips(used_clips, method="compose")

    print("[compose_segment_with_searcher] FIN")
    return final_clip
