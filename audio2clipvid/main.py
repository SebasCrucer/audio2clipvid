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
        llm_function (Callable[[str], str]): Función que recibe texto en lenguaje natural y retorna un string respuesta.
        embeddings_folder (str): Ruta donde se encuentran los embeddings generados (para ClipVidSearcher).
        videos_folder (str): Ruta donde se encuentran los videos (para ClipVidSearcher).
        output_video_path (str): Ruta de salida para el video final.
        device (str, optional): Dispositivo para WhisperX y para CLIP ("cuda" o "cpu"). Por defecto, detecta automáticamente.
        max_segment_words (int, optional): Máximo de palabras por segmento. Por defecto, 40.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Transcribir audio con WhisperX (obtener timestamps por palabra)
    transcribed_text, word_timestamps = transcribe_audio_with_whisperx(audio_path, device=device)

    # 2. Con la función lambda (llm_function), fragmentar el script
    splitted_fragments = split_script_into_fragments(transcribed_text, llm_function, max_words=max_segment_words)

    # 3. Generar descripciones cortas para cada fragmento
    short_descriptions = []
    for fragment in splitted_fragments:
        prompt_desc = f"Genera una breve descripción en inglés de hasta 5 palabras clave para: {fragment}"
        desc = llm_function(prompt_desc)
        short_descriptions.append(desc.strip())

    # 4. Buscar videos con ClipVidSearcher y crear un timeline
    if not ClipVidSearcher:
        raise ImportError("ClipVidSearcher no está disponible. Instala clipvid o usa tu propio buscador.")
    searcher = ClipVidSearcher(embeddings_folder=embeddings_folder, videos_folder=videos_folder, device=device)

    # Calcular rangos de tiempo para cada fragmento
    fragment_time_ranges = compute_fragment_time_ranges(word_timestamps, splitted_fragments)

    # 5. Para cada fragmento y descripción, componer clips y crear la lista de VideoClips
    segment_clips = []
    used_videos = set()  # si quieres evitar repetir videos, guárdalos en este set
    for (frag, desc), (start, end) in zip(zip(splitted_fragments, short_descriptions), fragment_time_ranges):
        duration = end - start
        clip_segment = compose_segment_with_searcher(
            searcher=searcher,
            description=desc,
            duration=duration,
            used_videos=used_videos,  # pasarle el set si deseas evitar repetidos
        )
        segment_clips.append(clip_segment)

    # 6. Concatenar todos los clips
    final_video_clip = mp.concatenate_videoclips(segment_clips, method="compose")

    # 7. Agregar el audio original
    audio_clip = mp.AudioFileClip(audio_path)

    # Si por algún motivo el total de los videos excede la duración del audio,
    # podemos forzar a que el clip final coincida con la duración del audio:
    if final_video_clip.duration > audio_clip.duration:
        final_video_clip = final_video_clip.subclip(0, audio_clip.duration)

    # Asignar el audio ya recortado
    final_video_clip = final_video_clip.set_audio(audio_clip)

    # 8. Guardar el video
    final_video_clip.write_videofile(output_video_path, fps=24, codec="libx264", audio_codec="aac")

    # Cerrar
    final_video_clip.close()
    audio_clip.close()
    for c in segment_clips:
        c.close()

    print(f"Video generado en: {output_video_path}")


def transcribe_audio_with_whisperx(audio_path: str, device: str = "cpu"):
    """
    Transcribe el audio usando WhisperX y retorna:
      - El texto completo transcrito
      - Una lista con (palabra, start_time, end_time)
    """
    model = whisperx.load_model("small", device=device)
    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device,
                                    return_char_alignments=False)

    # Reconstruir el texto completo y extraer timestamps palabra por palabra
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
    de un cierto tamaño (en palabras). Ajusta según tu lógica.
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
        fragments: List[str]
):
    """
    Dado el array (palabra, start, end) y la lista de fragmentos,
    asigna rangos de tiempo a cada fragmento (start_time, end_time).
    """
    fragment_time_ranges = []
    index_palabra = 0
    for fragment in fragments:
        fragment_word_count = len(fragment.split())
        fragment_start_time = word_timestamps[index_palabra][1]  # start
        fragment_end_time = word_timestamps[index_palabra + fragment_word_count - 1][2]  # end
        fragment_time_ranges.append((fragment_start_time, fragment_end_time))
        index_palabra += fragment_word_count

    return fragment_time_ranges


from moviepy.video.fx import Resize, Crop

def ajustar_clip_vertical(clip, target_w=1080, target_h=1920):
    cw, ch = clip.size
    target_ratio = target_w / target_h
    clip_ratio = cw / ch

    if clip_ratio > target_ratio:
        # Escalamos por alto
        scaled_clip = clip.fx(Resize, height=target_h)
        new_w, new_h = scaled_clip.size
        # Recortamos ancho sobrante
        x_center = new_w / 2
        x1 = x_center - (target_w / 2)
        x2 = x_center + (target_w / 2)
        final_clip = scaled_clip.crop(x1=x1, y1=0, x2=x2, y2=new_h)

    else:
        # Escalamos por ancho
        scaled_clip = clip.fx(Resize, width=target_w)
        new_w, new_h = scaled_clip.size
        if new_h > target_h:
            # Recortamos alto sobrante
            y_center = new_h / 2
            y1 = y_center - (target_h / 2)
            y2 = y_center + (target_h / 2)
            final_clip = scaled_clip.crop(x1=0, y1=y1, x2=new_w, y2=y2)
        else:
            # Si quedó pequeño, estirar forzado
            final_clip = scaled_clip.fx(Resize, newsize=(target_w, target_h))

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
    Usa el buscador ClipVidSearcher para encontrar un video o varios
    que coincidan con la `description`. Si no alcanza la duración necesaria,
    se concatenan varios resultados hasta cubrir la `duration`.

    Además:
      - Si used_videos está presente, evitamos reutilizar el mismo video.
      - Ajustamos cada clip a ratio 9:16 con `ajustar_clip_vertical`.
      - Recortamos el último clip si excede 'duration'.
      - Si no se encuentra nada válido, devolvemos un ColorClip de relleno.
    """
    if used_videos is None:
        used_videos = set()

    results = searcher.search(description, n=10)
    # results: [(similarity, index, video_path), ...]

    used_clips = []
    accumulated_duration = 0.0
    idx = 0

    while accumulated_duration < duration and idx < len(results):
        _, _, video_path = results[idx]
        idx += 1

        # Si ya se usó este video, saltar (opcional)
        if video_path in used_videos:
            continue

        if not os.path.exists(video_path):
            continue

        # Registrar como usado si queremos evitar duplicados
        used_videos.add(video_path)

        clip = mp.VideoFileClip(video_path)

        # Ajustar a 1080x1920
        clip = ajustar_clip_vertical(clip, target_w, target_h)

        clip_needed = duration - accumulated_duration
        if clip.duration > clip_needed:
            # recortar la parte que nos falta
            clip = clip.subclip(0, clip_needed)

        used_clips.append(clip)
        accumulated_duration += clip.duration

    # Si no se encontró nada o no se pudo cubrir la duración
    if not used_clips:
        filler_clip = mp.ColorClip(size=(target_w, target_h), color=(0, 0, 0), duration=duration)
        return filler_clip

    # Concatenamos
    if len(used_clips) == 1:
        final_clip = used_clips[0]
    else:
        final_clip = mp.concatenate_videoclips(used_clips, method="compose")

    return final_clip
