# Development checks

The v0.3.3 development suite contains 123 Python tests and 25 JavaScript/DOM tests. It covers planning, shot participation, localization, prompt construction, reference limits, timeline persistence, original/mute audio, resolution, frame timing, revisions and output assembly.

Run Python tests using the ComfyUI environment:

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

Run frontend tests with Node.js and Playwright:

```bash
npm install
npx playwright install chromium
npm test
```

On Windows the browser tests use the installed Microsoft Edge browser. `CSD_NODE_MODULES` may point to an existing Node modules directory.

Tests use temporary or synthetic media. They do not download model weights or queue a production render. Automated checks verify software behavior; evaluate generated videos using the actual source material and chosen generation settings.

Build a distributable ZIP and checksums with standard Python 3.11+:

```bash
python tools/build_release.py
```
