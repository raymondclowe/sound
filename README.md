# Sound Wake Word Detection Project

## Project Status (as of November 27, 2025)

This project is a Python-based wake word detection and speech transcription tool. It uses:
- Real-time audio capture and silence detection
- MFCC-based word matching for wake word detection
- Integration with external speech-to-text (STT) engines (mini_transcriber and Whisper.cpp)

### Current Features
- **Audio Buffering & Silence Detection**: Efficiently captures and segments audio for wake word detection.
- **MFCC Word Matching**: Uses Mel-Frequency Cepstral Coefficients to match spoken words against one or more reference wav files.
- **STT Integration**: Supports both mini_transcriber and Whisper.cpp endpoints for transcription.
- **Network Optimization**: Uses a cached IP address for `.local` hostnames to avoid repeated slow mDNS lookups.
- **Session Reuse**: HTTP requests use a persistent session for faster repeated calls.
- **Benchmarking Tools**: Includes utilities to benchmark DNS resolution and POST request times.

### How to Use the demo
1. Configure your reference word and run the demo. Record multiple people saying your target wake word.
2. The system will listen for the wake word and confirm with a chime when detected.
3. Transcription is performed using the selected STT backend.


Example commands:
```bash
# Ensure you have the real example WAVs in the root of the repository (not generated tones).

# Run demo (if using the repository directly)
python -m demo --debug

# After packaging or installation the console script is available:
easywakeword-demo --debug
```

### How to use in your own code

```python
from easywakeword import Recogniser

recogniser = Recogniser(
	wakewordstrings=["computer", "computer", "computer"],
	wakewordreferenceaudios=["examples/example_computer_male.wav", "examples/example_computer_female.wav", "examples/example_computer_male_teen.wav"],
	threshold=75,
	auto_sound_from_reference=True,
	stt_minscore=85.0,  # min score for STT
	stt_async=False,    # run STT synchronously by default
	allowed_other_words=["ok", "activate", "please", "hey"],
)

res = recogniser.waitforit()
if res:
	print("Detected:", res)
```


### Next Steps
- Further optimize audio preprocessing and detection.
- Add more robust error handling and logging.
- Explore additional STT backends or on-device models.

---

For more details, see the code and comments in `main.py`.

## Packaging & Publishing to PyPI

This project already uses a `pyproject.toml` file for packaging (PEP 621). Below is a checklist and a short guide to make this project pip-installable and publishable to PyPI.

Checklist
- Ensure `project.name` in `pyproject.toml` is unique on PyPI — consider `easywakeword` or similar.
- Update `project.version` and other fields (`authors`, `license`, `classifiers`, `urls`) in `pyproject.toml`.
- Ensure `readme = "README.md"` (already present) and `description` is filled.
- Create a license file (e.g., `LICENSE`).
- Ensure the package includes necessary files using `include` or `package-dir` configuration.
- Create tests and CI (GitHub Actions recommended) to run tests and build artifacts on push / tag.
- Build and test locally using `python -m build` and `twine check`.
- Upload to TestPyPI first, verify install, then upload to real PyPI.

Quick Publishing Steps
1. Install helpers: `pip install build twine`
2. Build distributions: `python -m build` (generates `dist/`)
3. Validate: `python -m twine check dist/*`
4. Upload to TestPyPI: `python -m twine upload --repository testpypi dist/*`
5. Test install from TestPyPI: `pip install --index-url https://test.pypi.org/simple/ PACKAGE_NAME`
6. When validated, upload to PyPI: `python -m twine upload dist/*`

Recommended `pyproject.toml` additions
- `project.license = { file = "LICENSE" }`
- `project.authors = [ { name = "Your Name", email = "you@example.com" } ]`
- `project.urls = { "Homepage" = "https://github.com/your/repo", "Documentation" = "https://" }`
- `classifiers = [ "License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3", ... ]`

CI / Release Automation
- Add a GitHub Actions workflow to build and deploy releases automatically on tag. Your workflow should:
	- Run lint and tests
	- Build sdist/wheel
	- Publish to TestPyPI if you want, then to real PyPI using the `pypa/gh-action-pypi-publish` action (recommended).

Troubleshooting
- Check `pip install -e .` locally before publishing.
- If your project name is already taken on PyPI, choose a different name in `pyproject.toml`.
- Ensure your dependencies have compatible version ranges for intended Python versions.

