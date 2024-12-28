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
        output_video_path: str,
        device: str = None
):
    """
    Convierte un audio en un video usando WhisperX para extraer timestamps por palabra,
    una función (lambda) para fragmentar el texto y generar descripciones, y ClipVidSearcher
    para buscar y componer clips.

    Args:
        audio_path (str): Ruta al archivo de audio.
        llm_function (Callable[[str], str]): Función que recibe texto en lenguaje natural y retorna un string respuesta.
        embeddings_folder (str): Ruta donde se encuentran los embeddings generados (para ClipVidSearcher).
        output_video_path (str): Ruta de salida para el video final.
        device (str, optional): Dispositivo para WhisperX y para CLIP ("cuda" o "cpu"). Por defecto, detecta automáticamente.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Transcribir audio con WhisperX (obtener timestamps por palabra)
    transcribed_text, word_timestamps = transcribe_audio_with_whisperx(audio_path, device=device)

    # 2. Con la función lambda (llm_function), fragmentar el script
    #    Suponemos que llm_function puede recibir: "Dame fragmentos de 40 palabras aprox"
    #    y que devuelva trozos del guion. Ajusta la lógica a tu gusto.
    splitted_fragments = split_script_into_fragments(transcribed_text, llm_function)

    # 3. Generar descripciones cortas para cada fragmento
    short_descriptions = []
    for fragment in splitted_fragments:
        prompt_desc = f"Genera una breve descripción en inglés de hasta 5 palabras clave para: {fragment}"
        desc = llm_function(prompt_desc)
        short_descriptions.append(desc.strip())

    # 4. Buscar videos con ClipVidSearcher y crear un timeline
    #    Asumimos que la duración de cada fragmento se basa en sus timestamps en el audio original.
    #    Por simplicidad, usaremos la parte de `word_timestamps` que corresponde a cada fragmento.
    if not ClipVidSearcher:
        raise ImportError("ClipVidSearcher no está disponible. Instala clipvid o usa tu propio buscador.")
    searcher = ClipVidSearcher(embeddings_folder=embeddings_folder, device=device)

    # Emparejar cada fragmento con su rango de tiempo en el audio.
    # Por ejemplo, splitted_fragments[i] -> (start_time, end_time)  -> short_descriptions[i]
    fragment_time_ranges = compute_fragment_time_ranges(word_timestamps, splitted_fragments)

    # Buscar y componer video
    segment_clips = []
    for (frag, desc), (start, end) in zip(zip(splitted_fragments, short_descriptions), fragment_time_ranges):
        duration = end - start
        clip_segment = compose_segment_with_searcher(searcher, desc, duration)
        segment_clips.append(clip_segment)

    # 5. Concatenar todos los clips
    final_video_clip = mp.concatenate_videoclips(segment_clips)

    # 6. Agregar el audio original
    audio_clip = mp.AudioFileClip(audio_path)

    new_audioclip = mp.CompositeAudioClip([audio_clip, mp.AudioFileClip(audio_path).subclipped(0, final_video_clip.duration)])
    final_video_clip.audio = new_audioclip

    # 7. Guardar el video
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
    # Ejemplo sencillo: "Dame fragmentos de N palabras"
    prompt = (
        f"Por favor divide el siguiente texto en fragmentos de alrededor de {max_words} palabras, "
        f"intentando que cada fragmento represente una idea completa. Devuelve la lista de fragmentos, "
        f"cada uno en una nueva línea.\n\n"
        f"Texto:\n{full_text}"
    )

    response = llm_function(prompt)
    # Suponiendo que la lambda retorna un texto con cada fragmento en una línea, por ejemplo:
    # "Fragmento 1\nFragmento 2\n..."
    fragments = [line.strip() for line in response.split("\n") if line.strip()]
    return fragments


def compute_fragment_time_ranges(
        word_timestamps: List[Tuple[str, float, float]],
        fragments: List[str]
):
    """
    Dado el array (palabra, start, end) y la lista de fragmentos,
    asigna rangos de tiempo a cada fragmento (start_time, end_time).

    Este es un ejemplo simplificado: vamos tomando tantas palabras como
    componen cada fragmento (en conteo), y asignamos un rango en base a
    la primera y la última palabra de ese fragmento.
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


def compose_segment_with_searcher(
        searcher,
        description: str,
        duration: float
) -> mp.VideoClip:
    """
    Usa el buscador CLIPVidSearcher (searcher) para encontrar un video corto
    que coincida con la `description`. Si no alcanza la duración necesaria,
    se concatenan varios resultados hasta cubrir la `duration`.
    """
    results = searcher.search(description, n=10)
    # results es una lista de (nombre_archivo, score)
    # asumiendo que son paths reales o algo similar
    used_clips = []
    accumulated_duration = 0.0
    idx = 0

    while accumulated_duration < duration and idx < len(results):
        video_name, _ = results[idx]
        # Si `video_name` no es la ruta, ajusta la lógica para armar la ruta real
        # p.ej: video_path = os.path.join("videos", f"{video_name}.mp4")

        # Debug prints:
        print("Resultados CLIP:", video_name)

        video_path = video_name if video_name.endswith(".mp4") else f"{video_name}.mp4"
        print("Intentando cargar:", video_path)
        print("Existe?", os.path.exists(video_path))

        idx += 1

        if not os.path.exists(video_path):
            continue

        clip = mp.VideoFileClip(video_path)
        clip_needed = duration - accumulated_duration

        if clip.duration > clip_needed:
            # recortar
            clip = clip.subclipped(0, clip_needed)

        used_clips.append(clip)
        accumulated_duration += clip.duration

    if not used_clips:
        # Caso en que no encontró nada, genera un colorclip de relleno
        filler_clip = mp.ColorClip(size=(720, 1280), color=(0, 0, 0), duration=duration)
        return filler_clip

    if len(used_clips) == 1:
        return used_clips[0]
    else:
        return mp.concatenate_videoclips(used_clips)
