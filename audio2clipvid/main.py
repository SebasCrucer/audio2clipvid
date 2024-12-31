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
    Genera un video final a partir de un audio:
      1) Transcribe con WhisperX (palabras + timestamps).
      2) Usa una función LLM (llm_function) para dividir el texto en fragmentos coherentes.
      3) Ajusta los fragmentos para no exceder la cantidad de palabras detectadas por WhisperX.
      4) Asigna rangos de tiempo a cada fragmento (el último se extiende hasta el final del audio).
      5) Usa ClipVidSearcher para encontrar clips que cubran cada fragmento.
      6) Concantena los videos y agrega el audio original.

    Args:
        audio_path (str): Ruta del audio de entrada.
        llm_function (Callable[[str], str]): Función que recibe un prompt y retorna un texto (fragmentos).
        embeddings_folder (str): Carpeta con embeddings para ClipVidSearcher.
        videos_folder (str): Carpeta con videos para ClipVidSearcher.
        output_video_path (str): Ruta de salida del video final.
        device (str, opcional): "cpu" o "cuda". Por defecto detecta GPU si está disponible.
        max_segment_words (int, opcional): Máx. aproximado de palabras por fragmento.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Transcribir
    transcribed_text, word_timestamps = transcribe_audio_with_whisperx(audio_path, device=device)

    # 2. Dividir en fragmentos con LLM
    splitted_fragments = split_script_into_fragments(transcribed_text, llm_function, max_words=max_segment_words)

    # 3. Ajustar fragmentos para no exceder palabras de WhisperX
    splitted_fragments = adjust_fragments_to_whisperx(splitted_fragments, word_timestamps)

    # 4. Calcular tiempos de inicio y fin (el último fragmento llega hasta final del audio)
    audio_clip = mp.AudioFileClip(audio_path)
    fragment_time_ranges = compute_fragment_time_ranges(word_timestamps, splitted_fragments, audio_clip.duration)

    # 5. Generar descripción corta para cada fragmento
    short_descriptions = []
    for fragment in splitted_fragments:
        prompt = f"Genera una breve descripción en inglés (hasta 5 palabras clave) de: {fragment}"
        short_descriptions.append(llm_function(prompt).strip())

    # 6. Crear buscador de videos
    searcher = ClipVidSearcher(embeddings_folder=embeddings_folder, videos_folder=videos_folder, device=device)

    # 7. Para cada fragmento, componer clip
    segment_clips = []
    used_videos = set()
    for (frag, desc), (start, end) in zip(zip(splitted_fragments, short_descriptions), fragment_time_ranges):
        duration_needed = end - start
        segment_clip = compose_segment_with_searcher(searcher, desc, duration_needed, used_videos)
        segment_clips.append(segment_clip)

    # 8. Unir clips y agregar audio
    final_video_clip = mp.concatenate_videoclips(segment_clips, method="compose")
    final_video_clip.audio = audio_clip

    # 9. Exportar
    final_video_clip.write_videofile(
        output_video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )

    # 10. Cerrar
    final_video_clip.close()
    audio_clip.close()
    for sc in segment_clips:
        sc.close()


def transcribe_audio_with_whisperx(audio_path: str, device: str = "cpu"):
    """
    Transcribe un audio usando WhisperX y retorna:
      - Texto completo.
      - Lista de (palabra, start_time, end_time).
    """
    model = whisperx.load_model("small", device=device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    # Alinear palabras
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(
        result["segments"], align_model, metadata, audio_path, device, return_char_alignments=False
    )

    # Construir texto y timestamps
    word_timestamps = []
    all_text = []
    for seg in result_aligned["segments"]:
        for w in seg["words"]:
            word_timestamps.append((w["word"], w["start"], w["end"]))
            all_text.append(w["word"])

    return " ".join(all_text), word_timestamps


def split_script_into_fragments(
    full_text: str,
    llm_function: Callable[[str], str],
    max_words: int = 40
):
    """
    Usa la función LLM para dividir el texto en fragmentos coherentes,
    cada uno con ~max_words de referencia.
    """
    prompt = (
        f"Por favor divide el siguiente texto en fragmentos, cada uno con cerca de {max_words} palabras, "
        f"sin cortar ideas a la mitad. Devuelve cada fragmento en una nueva línea.\n\n{full_text}"
    )
    response = llm_function(prompt)
    return [line.strip() for line in response.split("\n") if line.strip()]


def adjust_fragments_to_whisperx(
    fragments: List[str],
    word_timestamps: List[Tuple[str, float, float]]
):
    """
    Si los fragmentos exceden la cantidad real de palabras (según WhisperX),
    recorta o elimina del final los fragmentos sobrantes.
    """
    total_whisperx_words = len(word_timestamps)
    splitted_tokens = [frag.split() for frag in fragments]
    all_llm_tokens = [token for tokens in splitted_tokens for token in tokens]
    difference = len(all_llm_tokens) - total_whisperx_words

    if difference <= 0:
        return fragments

    i = len(fragments) - 1
    while difference > 0 and i >= 0:
        tokens = fragments[i].split()
        if len(tokens) <= difference:
            fragments.pop(i)
            difference -= len(tokens)
            i -= 1
        else:
            kept_tokens = tokens[:-difference]
            fragments[i] = " ".join(kept_tokens)
            difference = 0

    return fragments


def compute_fragment_time_ranges(
    word_timestamps: List[Tuple[str, float, float]],
    fragments: List[str],
    audio_duration: float
):
    """
    Asigna (start_time, end_time) a cada fragmento:
      - start_time = inicio de la primera palabra del fragmento.
      - end_time = inicio de la primera palabra del siguiente fragmento,
                   o (si es el último) el final completo del audio.
    """
    fragment_time_ranges = []
    index_word = 0

    for i, fragment in enumerate(fragments):
        # Saltar fragmentos vacíos
        words_in_fragment = fragment.split()
        if not words_in_fragment:
            continue

        # Inicio de este fragmento
        start_time = word_timestamps[index_word][1]

        # Ver si hay fragmento siguiente
        next_index = i + 1
        if next_index < len(fragments):
            next_word_count = len(fragments[next_index].split())
            end_index = index_word + len(words_in_fragment)
            if end_index < len(word_timestamps):
                end_time = word_timestamps[end_index][1]
            else:
                # Si no hay más palabras, fin = última palabra
                end_time = word_timestamps[-1][2]
        else:
            # Último fragmento => se extiende hasta fin del audio
            end_time = audio_duration

        fragment_time_ranges.append((start_time, end_time))
        index_word += len(words_in_fragment)

    return fragment_time_ranges


def ajustar_clip_vertical(clip: mp.VideoFileClip, target_w=1080, target_h=1920):
    """
    Ajusta un clip al formato vertical (9:16), reescalando y recortando.
    """
    cw, ch = clip.size
    target_ratio = target_w / target_h
    clip_ratio = cw / ch

    if clip_ratio > target_ratio:
        clip = clip.resized(height=target_h)
        new_w, new_h = clip.size
        x_center = new_w / 2
        x1 = x_center - (target_w / 2)
        x2 = x_center + (target_w / 2)
        return clip.cropped(x1=x1, y1=0, x2=x2, y2=new_h)
    else:
        clip = clip.resized(width=target_w)
        new_w, new_h = clip.size
        if new_h > target_h:
            y_center = new_h / 2
            y1 = y_center - (target_h / 2)
            y2 = y_center + (target_h / 2)
            return clip.cropped(x1=0, y1=y1, x2=new_w, y2=y2)
        return clip.resized((target_w, target_h))


def compose_segment_with_searcher(
    searcher: ClipVidSearcher,
    description: str,
    duration: float,
    used_videos=None,
    target_w=1080,
    target_h=1920
):
    """
    Usa ClipVidSearcher para encontrar uno o varios videos que cumplan con `description`.
    Concantena clips hasta cubrir `duration`. Ajusta cada clip a 9:16.
    Si no encuentra nada, retorna un ColorClip negro.
    """
    if used_videos is None:
        used_videos = set()

    results = searcher.search(description, n=10)
    used_clips = []
    accumulated = 0.0
    i = 0

    while accumulated < duration and i < len(results):
        _, _, video_path = results[i]
        i += 1
        if video_path in used_videos or not os.path.exists(video_path):
            continue

        used_videos.add(video_path)
        clip = mp.VideoFileClip(video_path)
        clip = ajustar_clip_vertical(clip, target_w, target_h)

        needed = duration - accumulated
        if clip.duration > needed:
            clip = clip.subclipped(0, needed)

        used_clips.append(clip)
        accumulated += clip.duration

    if not used_clips:
        return mp.ColorClip(size=(target_w, target_h), color=(0, 0, 0), duration=duration)
    return used_clips[0] if len(used_clips) == 1 else mp.concatenate_videoclips(used_clips, method="compose")
