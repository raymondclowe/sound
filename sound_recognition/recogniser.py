import numpy as np
import sounddevice as sd
import time
import os
import requests
from .matching import WordMatcher
from .silence import SoundBuffer
from .transcription import transcribe_audio, resolve_stt_ip, STT_HOSTNAME, STT_PORT

class Recogniser:
    def __del__(self):
        # Destructor: ensure audio stream is stopped and closed, and STT session is closed
        try:
            sb = getattr(self, 'soundBuffer', None)
            if sb is not None:
                stream = getattr(sb, 'sd_stream', None)
                if stream is not None:
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[DEBUG] Exception during Recogniser cleanup: {e}")
        # Removed close_stt_session import and call (unknown symbol)
    def __init__(self,
                 wakewordstrings=None,
                 wakewordreferenceaudios=None,
                 wordsminmax=(1,3),
                 threshold=75,
                 whisperurl=None,
                 whispermodel="tiny",
                 languages=None,
                 timeout=30,
                 device=None,
                 debug=False,
                 debug_playback=False,
                 min_silence_before=0.3,
                 min_sound=0.15,
                 max_sound=1.5,
                 min_trailing_silence=0.15,
                 max_audio_duration=2.0,
                 allowed_other_words=None,
                 padding=0.05):
        self.target_words = wakewordstrings or []
        self.wordsminmax = wordsminmax
        self.threshold = threshold
        self.whisperurl = whisperurl
        self.whispermodel = whispermodel
        self.languages = languages
        self.timeout = timeout
        self.device = device
        self.debug = debug
        self.debug_playback = debug_playback
        self.min_silence_before = min_silence_before
        self.min_sound = min_sound
        self.max_sound = max_sound
        self.min_trailing_silence = min_trailing_silence
        self.max_audio_duration = max_audio_duration
        self.allowed_other_words = allowed_other_words if allowed_other_words is not None else []
        self.padding = padding
        self.matcher = WordMatcher(sample_rate=16000)
        # Load multiple reference audios
        if wakewordreferenceaudios:
            for i, ref in enumerate(wakewordreferenceaudios):
                name = (wakewordstrings[i] if wakewordstrings and i < len(wakewordstrings) else f"target_{i+1}")
                self.matcher.load_reference_from_file(ref, word_name=name)
        # Cache STT IP once
        from .transcription import resolve_stt_ip, STT_PORT
        stt_ip = resolve_stt_ip()
        self.stt_url = f"http://{stt_ip}:{STT_PORT}"
        self.soundBuffer = SoundBuffer(device=device)

    def waitforit(self):
        state = 'waiting'
        silence_start_time = None
        sound_start_time = None
        sound_end_time = None
        while True:
            sd.sleep(100)
            is_currently_silent = self.soundBuffer.is_silent()
            current_time = time.time()
            if self.debug:
                print(f"[DEBUG] State: {state}, SilenceThresh: {getattr(self.soundBuffer, 'silence_threshold', None):.4f}")
            prev_state = state
            if state == 'waiting':
                if is_currently_silent:
                    state = 'in_silence'
                    silence_start_time = current_time
            elif state == 'in_silence':
                if not is_currently_silent and silence_start_time is not None:
                    silence_duration = current_time - silence_start_time
                    if silence_duration >= self.min_silence_before:
                        state = 'in_sound'
                        sound_start_time = current_time
                    else:
                        state = 'waiting'
            elif state == 'in_sound':
                if not is_currently_silent and sound_start_time is not None:
                    sound_duration = current_time - sound_start_time
                    if sound_duration > self.max_sound:
                        state = 'waiting'
                elif sound_start_time is not None:
                    sound_duration = current_time - sound_start_time
                    if self.min_sound <= sound_duration <= self.max_sound:
                        state = 'after_sound'
                        sound_end_time = current_time
                    else:
                        state = 'waiting'
            elif state == 'after_sound':
                if is_currently_silent and sound_end_time is not None:
                    trailing_silence_duration = current_time - sound_end_time
                    if trailing_silence_duration >= self.min_trailing_silence and sound_start_time is not None:
                        extract_start = sound_start_time - current_time - self.padding
                        extract_end = sound_end_time - current_time + self.padding
                        word_samples_with_padding = self.soundBuffer.return_last_n_seconds(abs(extract_start))
                        word_end_idx = int((abs(extract_end)) * SoundBuffer.FREQUENCY)
                        word_audio = word_samples_with_padding[:len(word_samples_with_padding) - word_end_idx]
                        audio_duration = len(word_audio) / SoundBuffer.FREQUENCY
                        if audio_duration > self.max_audio_duration:
                            if self.debug:
                                print(f"[Skipped: {audio_duration:.1f}s too long]")
                            state = 'waiting'
                            continue
                        best_name, best_score, all_scores = self.matcher.best_match(word_audio, threshold=self.threshold)
                        audio_rms = np.sqrt(np.mean(word_audio**2))
                        audio_max = np.max(np.abs(word_audio))
                        print(f"[Best match: {best_name} {best_score:.1f} | All: {all_scores} | RMS={audio_rms:.4f} Max={audio_max:.4f}]", end=" ")
                        if best_score >= self.threshold:
                            if self.debug_playback:
                                sd.play(word_audio, SoundBuffer.FREQUENCY)
                                sd.wait()
                            prompt = f"We are looking for the Wake word alone or in a short phrase, the wakeword today is : {best_name}" if best_name else "Wake word: computer"
                            transcription = transcribe_audio(word_audio, SoundBuffer.FREQUENCY, stt_url=self.stt_url, prompt=prompt, model=self.whispermodel)
                            if transcription:
                                import re
                                words = [re.sub(r'[^a-z0-9]', '', w) for w in transcription.strip().lower().rstrip('.,!?;:').split()]
                                words = [w for w in words if w]
                                word_count = len(words)
                                min_words, max_words = self.wordsminmax if isinstance(self.wordsminmax, (list, tuple)) else (1, 1)
                                if min_words <= word_count <= max_words:
                                    target = re.sub(r'[^a-z0-9]', '', best_name.lower()) if best_name else None
                                    if target and target in words:
                                        others = [w for w in words if w != target]
                                        if all(w in self.allowed_other_words for w in others):
                                            print(f"✓ CONFIRMED: '{transcription}'")
                                            return transcription.strip().lower().rstrip('.,!?;:')
                                        else:
                                            print(f"✗ REJECTED: '{transcription}' (unapproved extra words: {[w for w in others if w not in self.allowed_other_words]})")
                                    else:
                                        print(f"? DETECTED: '{transcription}' - expected '{best_name}'")
                                else:
                                    print(f"✗ REJECTED: '{transcription}' (word count {word_count} not in range {min_words}-{max_words})")
                            else:
                                print(f"? MATCH - STT failed")
                        else:
                            if self.debug_playback:
                                sd.play(word_audio, SoundBuffer.FREQUENCY)
                                sd.wait()
                        state = 'waiting'
            if self.debug and prev_state != state:
                print(f"[DEBUG] State transition: {prev_state} -> {state}")
        return None
