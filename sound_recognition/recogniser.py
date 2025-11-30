import numpy as np
import sounddevice as sd
import time
import os
import requests
from .matching import WordMatcher
from .silence import SoundBuffer
from .transcription import transcribe_audio, resolve_stt_ip, STT_HOSTNAME, STT_PORT

class Recogniser:
    def __init__(self,
                 wakewordstring=None,
                 wakewordreferenceaudio=None,
                 wordsminmax=(1,3),
                 threshold=75,
                 whisperurl=None,
                 whispermodel="tiny",
                 languages=None,
                 timeout=30,
                 device=None,
                 debug=False,
                 debug_playback=False):
        self.wakewordstring = wakewordstring
        self.wakewordreferenceaudio = wakewordreferenceaudio
        self.wordsminmax = wordsminmax
        self.threshold = threshold
        self.whisperurl = whisperurl or f"http://{resolve_stt_ip()}:{STT_PORT}"
        self.whispermodel = whispermodel
        self.languages = languages or ['en']
        self.timeout = timeout
        self.device = device
        self.debug = debug
        self.debug_playback = debug_playback
        self.soundBuffer = SoundBuffer(seconds=10, device=device, debug=debug)
        self.matcher = WordMatcher(sample_rate=SoundBuffer.FREQUENCY)
        self.target_word = None
        self._setup_reference()

    def _setup_reference(self):
        if self.wakewordreferenceaudio and os.path.exists(self.wakewordreferenceaudio):
            self.matcher.load_reference_from_file(self.wakewordreferenceaudio)
            import soundfile as sf
            ref_audio, _ = sf.read(self.wakewordreferenceaudio)
            self.target_word = transcribe_audio(ref_audio, SoundBuffer.FREQUENCY)
            if self.target_word:
                self.target_word = self.target_word.strip().lower().rstrip('.,!?;:')
        elif self.wakewordstring:
            # Optionally support text-only reference (future)
            pass
        else:
            raise ValueError("Must provide wakewordreferenceaudio or wakewordstring")

    def waitforit(self):
        # Wait for buffer to fill
        while self.soundBuffer.samples_collected < self.soundBuffer.buffer_length:
            sd.sleep(100)
        
        # Check initial state
        if self.soundBuffer.is_silent():
            state = 'in_silence'
            silence_start_time = time.time()
        else:
            state = 'waiting'
            silence_start_time = None
            
        sound_start_time = None
        sound_end_time = None
        
        while True:
            sd.sleep(100)
            is_currently_silent = self.soundBuffer.is_silent()
            current_time = time.time()
            
            if state == 'waiting':
                if is_currently_silent:
                    state = 'in_silence'
                    silence_start_time = current_time
                    
            elif state == 'in_silence':
                if not is_currently_silent:
                    silence_duration = current_time - silence_start_time
                    if silence_duration >= 1.0:
                        state = 'in_sound'
                        sound_start_time = current_time
                    else:
                        state = 'waiting'
                        
            elif state == 'in_sound':
                if not is_currently_silent:
                    sound_duration = current_time - sound_start_time
                    if sound_duration > 1.5:
                        state = 'waiting'
                else:
                    sound_duration = current_time - sound_start_time
                    if 0.5 <= sound_duration <= 1.5:
                        state = 'after_sound'
                        sound_end_time = current_time
                    else:
                        state = 'waiting'
                        
            elif state == 'after_sound':
                if is_currently_silent:
                    trailing_silence_duration = current_time - sound_end_time
                    if trailing_silence_duration >= 0.5:
                        # Extract the word
                        padding = 0.05
                        extract_start = sound_start_time - current_time - padding
                        extract_end = sound_end_time - current_time + padding
                        
                        word_samples_with_padding = self.soundBuffer.return_last_n_seconds(abs(extract_start))
                        word_end_idx = int((abs(extract_end)) * SoundBuffer.FREQUENCY)
                        word_audio = word_samples_with_padding[:len(word_samples_with_padding) - word_end_idx]
                        
                        audio_duration = len(word_audio) / SoundBuffer.FREQUENCY
                        if audio_duration > 3.0:
                            if self.debug:
                                print(f"[Skipped: {audio_duration:.1f}s too long]")
                            state = 'waiting'
                            continue
                        
                        matches, similarity = self.matcher.matches(word_audio, threshold=self.threshold)
                        
                        audio_rms = np.sqrt(np.mean(word_audio**2))
                        audio_max = np.max(np.abs(word_audio))
                        
                        # Calculate log-scale score: heavily emphasizes differences near 100%
                        # Much more aggressive scaling to separate good from bad matches
                        if similarity >= 100:
                            log_score = 100.0
                        else:
                            # Aggressive exponential scale: 96% should score around 50-60
                            # Formula: 100 - (100 - similarity)^2
                            log_score = 100 - (100 - similarity) ** 2.5
                        
                        print(f"[Match: {similarity:.1f}% LogScore: {log_score:.1f} (threshold={self.threshold}%) RMS={audio_rms:.4f} Max={audio_max:.4f}]", end=" ")
                        
                        if matches:
                            if self.debug_playback:
                                sd.play(word_audio, SoundBuffer.FREQUENCY)
                                sd.wait()
                            prompt = f"Wake word: {self.target_word}" if self.target_word else "Wake word: computer"
                            transcription = transcribe_audio(word_audio, SoundBuffer.FREQUENCY, prompt=prompt, model=self.whispermodel)
                            if transcription:
                                transcription_clean = transcription.strip().lower().rstrip('.,!?;:')
                                if self.target_word and self.target_word in transcription_clean.split():
                                    print(f"✓ CONFIRMED: '{transcription}'")
                                    return transcription_clean
                                else:
                                    print(f"? DETECTED: '{transcription}' - expected '{self.target_word}'")
                            else:
                                print(f"? MATCH - STT failed")
                        else:
                            if self.debug_playback:
                                sd.play(word_audio, SoundBuffer.FREQUENCY)
                                sd.wait()
                        state = 'waiting'
                else:
                    state = 'waiting'
