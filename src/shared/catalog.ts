import type { CatalogApp } from './types'

// The apps in the suite. Screenshots, icons and a README snapshot for each live in catalog/<id>/
// (refresh the README copies with `npm run sync-catalog`).

export const CATALOG: CatalogApp[] = [
  {
    id: 'rulz-trainer',
    name: 'RulzTrainer',
    tagline: 'LoRA trainer for diffusion models, with a guided WebUI',
    description:
      'Train LoRAs for MOSS-Image, Bria FIBO 2.0 and Krea 2 from your browser. A step-by-step wizard recommends ' +
      'every setting from your GPU memory and dataset, and the finished LoRAs load straight into ComfyUI.',
    category: 'AI Training',
    repo: 'speedyrulz/RulzTrainer',
    branch: 'main',
    install: {
      kind: 'source',
      setup: [{ label: 'Installing the Python environment and CUDA PyTorch', run: ['install.bat'] }]
    },
    launch: { target: 'run_webui.bat' },
    requirements: [
      { id: 'python', note: 'Python 3.10 to 3.12' },
      { id: 'nvidia', note: 'CUDA GPU; 16 GB of VRAM or more recommended' },
      { id: 'git', optional: true, note: 'Lets the launcher update it with git pull' }
    ],
    folderName: 'RulzTrainer',
    notes: [
      'Setup downloads CUDA PyTorch and the training libraries (about 4 GB).',
      'Base models (about 30 GB each) are downloaded later from the WebUI, into this folder.',
      'Launch opens a console window and the WebUI in your browser. Closing the console stops it.'
    ],
    highlights: ['MOSS-Image, FIBO 2.0, Krea 2', 'Wizard with recommended settings', 'Multi-GPU', 'ComfyUI-ready LoRAs']
  },
  {
    id: 'image-organizer',
    name: 'RulZ Image Organizer',
    tagline: 'Finds and groups the faces in your photo library, fully offline',
    description:
      'A photo library that works like Google Photos’ People view. Add folders and it detects every face, groups ' +
      'them by person, and lets you search by person, expression, head direction and body pose. Nothing leaves your PC.',
    category: 'Photos',
    repo: 'speedyrulz/RulZ-Image-Organizer',
    branch: 'main',
    install: { kind: 'zip', asset: '^RulZ-Image-Organizer-v[\\d.]+-win64\\.zip$' },
    launch: { target: 'RulZ Image Organizer.exe' },
    requirements: [],
    folderName: 'RulZ Image Organizer',
    notes: ['Library data is kept in %LOCALAPPDATA%\\RulZ Image Organizer, separate from the program.'],
    highlights: ['Face grouping', 'Search by expression & pose', 'HEIC support', 'Runs offline']
  },
  {
    id: 'image-tournament',
    name: 'RulZ Image Tournament',
    tagline: 'Finds your favorite image, two at a time',
    description:
      'A single-elimination tournament for your images. Compare two side by side or in a split view with synced ' +
      'zoom and pan, keep one, and the other goes to a Losers folder until one champion is left.',
    category: 'Photos',
    repo: 'speedyrulz/RulZ-Image-Tournament',
    branch: 'main',
    install: { kind: 'zip', asset: '^RulZ-Image-Tournament-v[\\d.]+-win64\\.zip$' },
    launch: { target: 'RulZ Image Tournament.exe' },
    requirements: [{ id: 'dotnet-fx' }],
    folderName: 'RulZ Image Tournament',
    detect: ['%LOCALAPPDATA%\\Programs\\RulZ Image Tournament\\RulZ Image Tournament.exe'],
    notes: ['A single .exe. Settings and tournament progress are kept in %APPDATA%\\RulZ Image Tournament.'],
    highlights: ['Side by side & split view', 'Synced zoom and pan', 'Undo', 'Resumes where you left off']
  },
  {
    id: 'llm-suite',
    name: 'RulZ LLM Suite',
    tagline: 'Run local LLMs with llama.cpp: models, chat, tools and an API server',
    description:
      'A desktop app for running local LLMs in your own llama.cpp build. Browse your GGUF models, load them with a ' +
      'memory estimate, chat with streaming and reasoning blocks, use MCP tools, and serve OpenAI- and ' +
      'Anthropic-compatible APIs.',
    category: 'LLM',
    repo: 'speedyrulz/rulz-llm-suite',
    branch: 'master',
    install: {
      kind: 'installer',
      asset: '^RulZ-LLM-Suite-Setup-[\\d.]+\\.exe$',
      uninstallName: 'RulZ LLM Suite',
      shortcutName: 'RulZ LLM Suite'
    },
    launch: { target: 'RulZ LLM Suite.exe' },
    requirements: [{ id: 'nvidia', optional: true, note: 'For GPU offload in llama.cpp' }],
    folderName: 'RulZ LLM Suite',
    notes: [
      'Needs a llama.cpp build with llama-server.exe; set its folder in the app under Settings ▸ Engine.',
      'Installs with its own setup program (silently), and can be removed from Windows Settings too.'
    ],
    highlights: ['GGUF model library', 'Chat with tools (MCP)', 'Speculative decoding', 'Local OpenAI/Anthropic API']
  },
  {
    id: 'training-captioner',
    name: 'Rulz Training Captioner',
    tagline: 'Captions image datasets with a local vision LLM',
    description:
      'Caption training datasets with a local vision model. A thumbnail grid with an editable caption under every ' +
      'image, saved as .txt files next to them. Runs llama.cpp across one or more GPUs, or uses any ' +
      'OpenAI-compatible server.',
    category: 'AI Training',
    repo: 'speedyrulz/Rulz-Training-Captioner',
    branch: 'main',
    install: {
      kind: 'source',
      setup: [
        { label: 'Creating the Python environment', run: ['python', '-m', 'venv', '.venv'] },
        {
          label: 'Installing PySide6 and requests',
          run: ['.venv\\Scripts\\python.exe', '-m', 'pip', 'install', '-r', 'requirements.txt']
        }
      ]
    },
    launch: { target: '.venv\\Scripts\\pythonw.exe', args: ['main.py'], icon: 'assets\\icon.ico' },
    requirements: [
      { id: 'python', note: 'Python 3.10 or newer' },
      { id: 'nvidia', optional: true, note: 'For managed llama.cpp mode' }
    ],
    components: [
      {
        id: 'llama-cpp',
        label: 'llama.cpp (CUDA 13 build)',
        description: 'The llama-server the app starts in managed mode. Skip it if you use LM Studio, Ollama or vLLM.',
        sizeHint: 'about 700 MB',
        default: true,
        repo: 'ggml-org/llama.cpp',
        assets: ['^llama-b\\d+-bin-win-cuda-13\\.\\d+-x64\\.zip$', '^cudart-llama-bin-win-cuda-13\\.\\d+-x64\\.zip$'],
        dest: 'llama.cpp'
      }
    ],
    folderName: 'Rulz Training Captioner',
    notes: ['You also need a vision-capable GGUF model and its mmproj file (Qwen-VL, Gemma 3, MiniCPM-V…).'],
    highlights: ['Editable caption grid', 'Multi-GPU parallel or split', 'Custom prompts', 'Any OpenAI-compatible server']
  },
  {
    id: 'clip-extractor',
    name: 'RulZ Simple Video Clip Extractor',
    tagline: 'Frame-accurate clip extraction with lossless cuts and NVENC export',
    description:
      'A small video trimmer: step frame by frame or keyframe by keyframe, mark In and Out, and export losslessly or ' +
      're-encoded with NVIDIA NVENC. Crop, rotate, fixed-length clips and skin-tone colour matching included.',
    category: 'Video',
    repo: 'speedyrulz/rulz-simple-video-clip-extractor',
    branch: 'main',
    install: { kind: 'zip', asset: '^RulZ-Simple-Video-Clip-Extractor-v[\\d.]+-win64\\.zip$' },
    launch: { target: 'RulZ Clip Extractor.exe' },
    requirements: [
      { id: 'ffmpeg', note: 'ffmpeg and ffprobe on PATH' },
      { id: 'nvidia', optional: true, note: 'For NVENC export' }
    ],
    folderName: 'RulZ Clip Extractor',
    highlights: ['Frame-accurate', 'Lossless cuts', 'NVENC export', 'Crop & colour match']
  }
]

export function findApp(id: string): CatalogApp | undefined {
  return CATALOG.find((a) => a.id === id)
}
