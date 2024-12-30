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
    una función (lambda) para fragmentar el texto y generar descripciones, y ClipVidSearcher
    para buscar y componer clips.

    Args:
        audio_path (str): Ruta al archivo de audio.
        llm_function (Callable[[str], str]): Función LLM que recibe texto y retorna un string de respuesta.
        embeddings_folder (str): Ruta donde se encuentran los embeddings generados (para ClipVidSearcher).
        videos_folder (str): Ruta donde se encuentran los videos (para ClipVidSearcher).
        output_video_path (str): Ruta de salida para el video final.
        device (str, optional): "cuda" o "cpu". Si no se especifica, lo detecta.
        max_segment_words (int, optional): Máximo de palabras por fragmento. Por defecto, 40.
    """

    # 1. Determinar dispositivo
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Transcribir audio con WhisperX
    transcribed_text, word_timestamps = transcribe_audio_with_whisperx(audio_path, device=device)

    # 3. Dividir el texto en fragmentos con llm_function
    splitted_fragments = split_script_into_fragments(transcribed_text, llm_function, max_words=max_segment_words)

    # 4. Cargar el audio completo y medir duración
    audio_clip = mp.AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    # 5. Calcular rangos de tiempo, añadiendo un fragmento FILLER si sobra audio
    fragment_time_ranges = compute_fragment_time_ranges(
        word_timestamps,
        splitted_fragments,
        audio_duration
    )

    # 6. Generar una descripción corta para cada fragmento (incluyendo el FILLER)
    short_descriptions = []
    for fragment in splitted_fragments:
        if fragment == "FILLER":
            # Podrías cambiar esta descripción si quieres otro tipo de relleno
            desc = "Silence or filler"
        else:
            prompt_desc = (
                f"Genera una breve descripción en inglés de hasta 5 palabras clave para: {fragment}"
            )
            desc = llm_function(prompt_desc)
        short_descriptions.append(desc.strip())

    # 7. Crear el buscador de videos con ClipVidSearcher
    searcher = ClipVidSearcher(
        embeddings_folder=embeddings_folder,
        videos_folder=videos_folder,
        device=device
    )

    # 8. Para cada fragmento y descripción, generar un clip que cubra su duración
    segment_clips = []
    used_videos = set()

    for (frag, desc), (start, end) in zip(zip(splitted_fragments, short_descriptions), fragment_time_ranges):
        required_duration = end - start
        # Generar el video que cubre este rango
        clip_segment = compose_segment_with_searcher(
            searcher=searcher,
            description=desc,
            duration=required_duration,
            used_videos=used_videos,  # Para no repetir videos
        )
        segment_clips.append(clip_segment)

    # 9. Concatenar los clips de todos los fragmentos
    final_video_clip = mp.concatenate_videoclips(segment_clips, method="compose")

    # 10. Asignar el audio original completo (NOTA: no recortamos audio, pues tenemos FILLER)
    final_video_clip.audio = audio_clip

    # 11. Exportar el video
    final_video_clip.write_videofile(
        output_video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )

    # 12. Cerrar recursos
    final_video_clip.close()
    audio_clip.close()
    for sc in segment_clips:
        sc.close()

    print(f"Video generado en: {output_video_path}")


def transcribe_audio_with_whisperx(audio_path: str, device: str = "cpu"):
    """
    Transcribe el audio usando WhisperX y retorna:
      - El texto completo transcrito
      - Una lista con (palabra, start_time, end_time)
    """
    # Cargar modelo
    model = whisperx.load_model("small", device=device)
    audio = whisperx.load_audio(audio_path)

    # Transcribir
    result = model.transcribe(audio)

    # Alinear
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio_path, device,
        return_char_alignments=False
    )

    # Reconstruir texto y extraer timestamps por palabra
    word_timestamps = []
    all_text = []
    for seg in result_aligned["segments"]:
        for w in seg["words"]:
            word_timestamps.append((w["word"], w["start"], w["end"]))
            all_text.append(w["word"])

    transcript_text = " ".join(all_text)
    return transcript_text, word_timestamps


def split_script_into_fragments(full_text: str, llm_function: Callable[[str], str], max_words: int = 40):
    """
    Usa la función llm_function para generar la división del texto en fragmentos
    de un cierto tamaño (en palabras).
    """
    prompt = (
        f"Por favor divide el siguiente texto en fragmentos de alrededor de {max_words} palabras, "
        f"intentando que cada fragmento represente una idea completa. Devuelve la lista de fragmentos, "
        f"cada uno en una nueva línea.\n\n"
        f"Texto:\n{full_text}"
    )
    response = llm_function(prompt)
    fragments = [line.strip() for line in response.split("\n") if line.strip()]
    return fragments


def compute_fragment_time_ranges(
        word_timestamps: List[Tuple[str, float, float]],
        fragments: List[str],
        audio_duration: float
):
    """
    Dado el array (palabra, start, end) y la lista de fragmentos,
    asigna rangos de tiempo a cada fragmento (start_time, end_time).

    Además, si sobra tiempo después de la última palabra, añadimos
    un fragmento "FILLER" para cubrir [última_palabra, fin_audio].
    """
    fragment_time_ranges = []
    index_palabra = 0

    # 1) Calcular los rangos de tiempo para fragmentos "hablados"
    for fragment in fragments:
        # Si en algún caso tienes un "FILLER" ya metido antes, evita error:
        if fragment == "FILLER":
            # Lo saltamos momentáneamente (esto se maneja más abajo)
            continue

        fragment_word_count = len(fragment.split())
        fragment_start_time = word_timestamps[index_palabra][1]  # start
        fragment_end_time = word_timestamps[index_palabra + fragment_word_count - 1][2]  # end

        fragment_time_ranges.append((fragment_start_time, fragment_end_time))
        index_palabra += fragment_word_count

    # 2) Determinar si hay más audio después de la última palabra
    if fragment_time_ranges:
        last_fragment_end = fragment_time_ranges[-1][1]
    else:
        # Caso extremo si no hay palabras
        last_fragment_end = 0.0

    # 3) Si la duración del audio es mayor al end de la última palabra...
    if audio_duration > last_fragment_end:
        # Añadimos un fragmento FILLER al final de la lista de "fragments"
        fragments.append("FILLER")
        fragment_time_ranges.append((last_fragment_end, audio_duration))

    return fragment_time_ranges


def ajustar_clip_vertical(clip: mp.VideoFileClip, target_w=1080, target_h=1920):
    """
    Ajusta un clip al formato vertical 1080x1920 (9:16).
    """
    cw, ch = clip.size
    target_ratio = target_w / target_h
    clip_ratio = cw / ch

    if clip_ratio > target_ratio:
        # El clip es más ancho que 9:16; escalamos por altura y recortamos a los lados
        scaled_clip = clip.resize(height=target_h)
        new_w, new_h = scaled_clip.size
        x_center = new_w / 2
        x1 = x_center - (target_w / 2)
        x2 = x_center + (target_w / 2)
        final_clip = scaled_clip.crop(x1=x1, y1=0, x2=x2, y2=new_h)
    else:
        # El clip es más alto o igual al ratio; escalamos por ancho y recortamos arriba/abajo si hace falta
        scaled_clip = clip.resize(width=target_w)
        new_w, new_h = scaled_clip.size
        if new_h > target_h:
            y_center = new_h / 2
            y1 = y_center - (target_h / 2)
            y2 = y_center + (target_h / 2)
            final_clip = scaled_clip.crop(x1=0, y1=y1, x2=new_w, y2=y2)
        else:
            # Si aún así queda menor, lo estiramos del todo
            final_clip = scaled_clip.resize((target_w, target_h))
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
    que coincidan con `description`. Se concatena todo lo necesario
    hasta cubrir `duration`.

    - Si used_videos existe, se evita reutilizar un video ya usado.
    - Ajusta cada clip a 9:16 con `ajustar_clip_vertical`.
    - Recorta el último clip si excede la duración necesaria.
    - Si no se encuentra nada, devuelve un ColorClip de relleno.
    """
    if used_videos is None:
        used_videos = set()

    # Buscamos top 10 videos
    results = searcher.search(description, n=10)
    # results es una lista de (similarity, index, video_path)

    used_clips = []
    accumulated_duration = 0.0
    idx = 0

    # Ir concatenando hasta cubrir la duración
    while accumulated_duration < duration and idx < len(results):
        _, _, video_path = results[idx]
        idx += 1

        # Si ya se usó, saltar
        if video_path in used_videos:
            continue

        # Marcarlo como usado
        used_videos.add(video_path)

        # Verificar que exista
        if not os.path.exists(video_path):
            continue

        clip = mp.VideoFileClip(video_path)
        clip = ajustar_clip_vertical(clip, target_w, target_h)

        clip_needed = duration - accumulated_duration
        if clip.duration > clip_needed:
            # Recortamos la parte que sobre
            clip = clip.subclip(0, clip_needed)

        used_clips.append(clip)
        accumulated_duration += clip.duration

    # Si no se encontró nada o no se cubrió la duración
    if not used_clips:
        # Relleno en negro
        filler_clip = mp.ColorClip(size=(target_w, target_h), color=(0, 0, 0), duration=duration)
        return filler_clip

    # Concatenar todos los subclips
    if len(used_clips) == 1:
        final_clip = used_clips[0]
    else:
        final_clip = mp.concatenate_videoclips(used_clips, method="compose")

    return final_clip
