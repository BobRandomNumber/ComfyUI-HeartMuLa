import os
import platform
import server
from aiohttp import web
from .nodes import (
    HeartMuLaLoader,
    HeartMuLaCodecLoader,
    HeartMuLaMusicGenerator,
    HeartMuLaAudioDecoder,
    HeartMuLaTranscriptionLoader,
    HeartMuLaLyricsTranscriber,
    HeartMuLaPostProcessor
)

@server.PromptServer.instance.routes.get("/heartmula/browse")
async def browse_directory(request):
    path = request.query.get("path", "")
    is_windows = platform.system() == "Windows"
    
    if not path or path == "root":
        if is_windows:
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
            return web.json_response({
                "current_path": "root",
                "dirs": drives,
                "parent": None,
                "is_windows_root": True
            })
        else:
            path = "/"

    if not os.path.exists(path):
        return web.json_response({"error": f"Path not found: {path}"}, status=404)
        
    try:
        items = os.listdir(path)
        dirs = []
        for d in items:
            try:
                full_item_path = os.path.join(path, d)
                if os.path.isdir(full_item_path):
                    dirs.append(d)
            except (PermissionError, OSError):
                continue
                
        parent = os.path.abspath(os.path.join(path, ".."))
        if is_windows and (parent == path or len(path) <= 3):
            parent = "root"
            
        return web.json_response({
            "current_path": os.path.abspath(path),
            "dirs": sorted(dirs),
            "parent": parent,
            "is_windows_root": False
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

NODE_CLASS_MAPPINGS = {
    "HeartMuLaLoader": HeartMuLaLoader,
    "HeartMuLaCodecLoader": HeartMuLaCodecLoader,
    "HeartMuLaMusicGenerator": HeartMuLaMusicGenerator,
    "HeartMuLaAudioDecoder": HeartMuLaAudioDecoder,
    "HeartMuLaTranscriptionLoader": HeartMuLaTranscriptionLoader,
    "HeartMuLaLyricsTranscriber": HeartMuLaLyricsTranscriber,
    "HeartMuLaPostProcessor": HeartMuLaPostProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLaLoader": "HeartMuLa Loader",
    "HeartMuLaCodecLoader": "HeartMuLa Codec Loader",
    "HeartMuLaMusicGenerator": "HeartMuLa Music Generator",
    "HeartMuLaAudioDecoder": "HeartMuLa Audio Decoder",
    "HeartMuLaTranscriptionLoader": "HeartMuLa Transcription Loader",
    "HeartMuLaLyricsTranscriber": "HeartMuLa Lyrics Transcriber",
    "HeartMuLaPostProcessor": "Audio Post-Processor",
}

WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]