"""Controlled 1-minute multi-shot proof-of-concept workflow for CineAI.

Defines a 30-shot (~60s at 2.0s/shot) fictional cinematic sequence inspired by
cosmic/timeless visual storytelling, and runs a controlled batch through
CineAI's VideoService (CogVideoX) and MovieAssembler without modifying backend
pipelines or database schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.services.movie_assembler import (  # noqa: E402
    MovieAssembler,
    MovieAssemblyRequest,
    MovieSceneAssets,
)
from app.services.video import VideoService  # noqa: E402
from app.services.video_providers import (  # noqa: E402
    CogVideoXVideoProvider,
    VideoGenerationRequest,
)


@dataclass(frozen=True)
class ShotSpec:
    shot_number: int
    scene_description: str
    camera_direction: str
    visual_prompt: str
    duration: float
    transition_information: Literal["cut", "fade"]


# 30-shot fictional cinematic test sequence (~60s total at 2.0s/shot).
# Inspired by epic cosmic-odyssey visual storytelling without hard-coding any
# specific story IP.
TEST_SEQUENCE_SHOTS: List[ShotSpec] = [
    ShotSpec(
        shot_number=1,
        scene_description="A lone wanderer stands on a windswept salt flat beneath a twin-moon twilight sky.",
        camera_direction="Wide establishing shot, slow forward dolly",
        visual_prompt="Cinematic wide shot of a lone traveler in a weathered cloak standing on reflective salt flats under a deep indigo twilight sky with faint nebula dust, slow forward dolly, atmospheric haze, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=2,
        scene_description="Ancient brass astrolabe rings begin rotating inside a sunlit sandstone ruin.",
        camera_direction="Medium close-up, gentle orbital pan right",
        visual_prompt="Cinematic medium close-up of interlocking weathered brass astrolabe rings slowly rotating in golden shafts of dust-filled sunlight inside ancient sandstone ruins, shallow depth of field, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=3,
        scene_description="A glowing blue geometric artifact pulses softly in the traveler's gloved hands.",
        camera_direction="Close-up, subtle handheld push-in",
        visual_prompt="Cinematic close-up of gloved hands holding a translucent sapphire polyhedron glowing softly with internal golden filaments against dark obsidian stone, subtle push-in, crisp film lighting, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=4,
        scene_description="Wind sweeps crimson dust across a colossal sunken temple courtyard.",
        camera_direction="High-angle crane descent",
        visual_prompt="Cinematic high-angle shot descending over massive carved stone pillars half-buried in crimson desert dunes as wind-blown sand drifts across the courtyard, dramatic dusk shadows, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=5,
        scene_description="Holographic star-chart filaments unfold in a darkened observatory chamber.",
        camera_direction="Medium shot, slow arc left",
        visual_prompt="Cinematic medium shot of glowing amber and cyan star-map filaments floating in mid-air inside a dark circular stone observatory, slow camera arc left, volumetric light, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=6,
        scene_description="Footsteps press into wet black volcanic sand along a mist-shrouded shore.",
        camera_direction="Low-angle tracking shot",
        visual_prompt="Cinematic low-angle tracking shot following boots stepping across wet black volcanic sand beside calm silver ocean waves shrouded in morning mist, moody natural lighting, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=7,
        scene_description="A monolithic obsidian arch rises above swirling clouds at dawn.",
        camera_direction="Low-angle tilt up",
        visual_prompt="Cinematic low-angle tilt up toward a towering smooth black obsidian archway piercing a sea of golden dawn clouds on a mountain ridge, epic scale, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=8,
        scene_description="Raindrops ripple across a dark mirror pool reflecting constellations.",
        camera_direction="Overhead top-down shot",
        visual_prompt="Cinematic top-down shot of gentle raindrop ripples spreading across a still black mirror pool reflecting bright starry constellations overhead, serene lighting, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=9,
        scene_description="An elder archivist turns fragile copper-inlaid pages by candlelight.",
        camera_direction="Medium close-up, static locked frame",
        visual_prompt="Cinematic medium close-up of weathered hands turning an ancient parchment manuscript etched with glowing copper diagrams under warm flickering candlelight, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=10,
        scene_description="A sleek silver research skiff glides silently over a canyon of glowing quartz.",
        camera_direction="Aerial tracking shot",
        visual_prompt="Cinematic aerial tracking shot of a compact silver exploration craft gliding smoothly above a deep canyon lined with bioluminescent violet quartz formations, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=11,
        scene_description="Dust motes drift around a suspended pendulum clock frozen mid-swing.",
        camera_direction="Macro detail push-in",
        visual_prompt="Cinematic macro shot pushing in toward an ornate brass pendulum suspended motionless in warm amber sunbeams with floating dust motes, shallow depth of field, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=12,
        scene_description="The traveler crosses a narrow rope-and-stone bridge suspended over glacial mist.",
        camera_direction="Wide side-profile tracking shot",
        visual_prompt="Cinematic wide shot tracking a cloaked figure walking steadily across an ancient stone footbridge suspended above deep blue glacial fog, crisp alpine sunlight, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=13,
        scene_description="Luminous golden spores drift upward through a sunken forest of petrified trees.",
        camera_direction="Medium wide shot, slow vertical pedestal up",
        visual_prompt="Cinematic medium wide shot rising slowly among towering pale petrified trees as glowing golden spores float upward through emerald mist, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=14,
        scene_description="A child watches paper lanterns rise toward an aurora-lit horizon.",
        camera_direction="Over-the-shoulder medium shot",
        visual_prompt="Cinematic over-the-shoulder shot of a child in a wool coat watching warm glowing paper lanterns ascend into an emerald aurora borealis night sky, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=15,
        scene_description="Liquid mercury forms concentric standing waves inside a stone basin.",
        camera_direction="Close-up, slow top-down descent",
        visual_prompt="Cinematic close-up descending toward concentric geometric ripples shimmering across liquid silver inside a carved basalt basin, dramatic rim lighting, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=16,
        scene_description="A silhouette approaches a blinding doorway of warm white light inside a cavern.",
        camera_direction="Medium wide shot, slow push-in",
        visual_prompt="Cinematic medium wide shot from behind a dark silhouette walking slowly toward a radiant rectangular portal of warm golden light at the end of a rough stone cavern, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=17,
        scene_description="Solar sails unfurl silently against the curve of a pale blue planet.",
        camera_direction="Orbital wide shot, slow roll right",
        visual_prompt="Cinematic orbital wide shot of thin golden solar sails unfolding in deep space above the luminous blue atmospheric horizon of a distant planet, crisp starlight, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=18,
        scene_description="Rain streaks down a glass observatory dome overlooking neon-lit rain streets.",
        camera_direction="Interior close-up, rack focus to city lights",
        visual_prompt="Cinematic close-up of rain droplets sliding down curved observatory glass with warm amber and cyan bokeh city lights glowing softly below at night, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=19,
        scene_description="A compass needle spins wildly before locking toward a distant mountain beacon.",
        camera_direction="Extreme close-up, subtle camera drift",
        visual_prompt="Cinematic extreme close-up of an antique brass compass resting on wet slate as its needle spins and locks onto a glowing distant mountain horizon, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=20,
        scene_description="White doves burst upward from a sunlit courtyard colonnade.",
        camera_direction="Medium wide shot, slight upward tilt",
        visual_prompt="Cinematic medium wide shot of white doves taking flight across a sunlit marble colonnade with long dramatic morning shadows, graceful slow-motion feel, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=21,
        scene_description="Fractured icebergs drift past a lone red lighthouse beacon in twilight.",
        camera_direction="Wide coastal aerial glide",
        visual_prompt="Cinematic aerial glide over dark arctic waters filled with drifting blue icebergs toward a solitary stone lighthouse casting a warm beam in deep twilight, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=22,
        scene_description="Golden sand pours upward in slow reverse motion inside a glass hourglass.",
        camera_direction="Close-up, locked studio macro",
        visual_prompt="Cinematic macro close-up of fine golden sand streams flowing gracefully upward inside an ornate glass hourglass bathed in warm rim lighting, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=23,
        scene_description="A circular stone dial aligns with a beam of eclipse sunlight.",
        camera_direction="Medium shot, slow dolly left",
        visual_prompt="Cinematic medium shot of a carved stone sundial illuminated by a sharp shaft of eclipse sunlight cutting through swirling temple dust, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=24,
        scene_description="Two figures stand apart on a windswept cliff overlooking an endless cloud sea.",
        camera_direction="Wide silhouette shot, gentle push-in",
        visual_prompt="Cinematic wide shot of two silhouettes standing on a rugged cliff edge looking out across an endless sea of sunset clouds glowing rose and gold, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=25,
        scene_description="Bioluminescent waves crash gently along a dark cavern beach.",
        camera_direction="Low-angle wide shot",
        visual_prompt="Cinematic low-angle shot of electric-blue bioluminescent ocean waves breaking softly on wet cavern rocks under faint shafts of moonlight, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=26,
        scene_description="Concentric rings of floating stone rise around an illuminated core.",
        camera_direction="Medium wide shot, steady push-in",
        visual_prompt="Cinematic medium wide shot of carved basalt rings levitating in slow counter-rotation around a warm radiant sphere inside a vast subterranean hall, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=27,
        scene_description="Morning mist clears over terraced green valleys and winding silver rivers.",
        camera_direction="High aerial flyover",
        visual_prompt="Cinematic aerial flyover emerging through parting white clouds to reveal lush emerald terraced hills and a winding silver river catching sunrise light, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=28,
        scene_description="Close-up of eyes reflecting a swirling golden spiral of starlight.",
        camera_direction="Extreme close-up, slow push-in",
        visual_prompt="Cinematic extreme close-up of human eyes illuminated by the warm reflection of swirling golden starlight and cosmic nebulae, crisp skin detail, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
    ShotSpec(
        shot_number=29,
        scene_description="A lone figure walks forward into a radiant horizon of endless white light.",
        camera_direction="Wide rear tracking shot",
        visual_prompt="Cinematic wide rear shot of a lone figure walking calmly along a mirror-like shallow water plain toward a glowing golden sunrise horizon, serene atmosphere, photorealistic.",
        duration=2.0,
        transition_information="fade",
    ),
    ShotSpec(
        shot_number=30,
        scene_description="Distant galaxy spiral shimmers softly as darkness settles into calm starlight.",
        camera_direction="Deep-space slow pull-back",
        visual_prompt="Cinematic deep-space shot pulling back gently from a luminous golden spiral galaxy surrounded by tranquil starfields and subtle violet cosmic dust, photorealistic.",
        duration=2.0,
        transition_information="cut",
    ),
]


def validate_mp4(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    expected_frames: int | None = None,
) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"MP4 missing or empty: {path}")

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        meta = reader.get_meta_data()
        frames = [frame for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"MP4 contains zero decodable frames: {path}")

    actual_size = meta.get("size")
    if actual_size != (expected_width, expected_height):
        raise RuntimeError(
            f"MP4 {path.name} resolution {actual_size} != {(expected_width, expected_height)}"
        )

    actual_fps = float(meta.get("fps", 0.0))
    if abs(actual_fps - expected_fps) > 0.1:
        raise RuntimeError(f"MP4 {path.name} FPS {actual_fps} != {expected_fps}")

    if expected_frames is not None and len(frames) != expected_frames:
        raise RuntimeError(
            f"MP4 {path.name} frame count {len(frames)} != {expected_frames}"
        )

    frame_arrays = [np.asarray(frame, dtype=np.float32) for frame in frames]
    frame_means = [float(arr.mean()) for arr in frame_arrays]
    diffs = [
        float(np.abs(frame_arrays[i + 1] - frame_arrays[i]).mean())
        for i in range(len(frame_arrays) - 1)
    ]
    if max(frame_means) <= 0.5:
        raise RuntimeError(f"MP4 {path.name} decoded to all-black frames.")

    return {
        "valid": True,
        "error": None,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "codec": meta.get("codec"),
        "pix_fmt": meta.get("pix_fmt"),
        "width": actual_size[0],
        "height": actual_size[1],
        "fps": actual_fps,
        "frame_count": len(frames),
        "duration_seconds": len(frames) / actual_fps,
        "mean_brightness": round(float(np.mean(frame_means)), 2),
        "mean_frame_diff": round(float(np.mean(diffs)) if diffs else 0.0, 2),
        "first_frame_mean": round(frame_means[0], 2),
        "last_frame_mean": round(frame_means[-1], 2),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_manifest_atomic(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now_iso()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def _initial_shot_entry(shot: ShotSpec, seed: int) -> Dict[str, Any]:
    return {
        **asdict(shot),
        "seed": seed,
        "output_path": None,
        "video_url": None,
        "generation_status": "pending",
        "generation_time": {
            "wall_time_seconds": None,
            "inference_time_seconds": None,
        },
        "validation_result": None,
        "gpu_peak_allocated_mib": None,
        "gpu_peak_reserved_mib": None,
        "attempts": 0,
        "error": None,
    }


def _load_or_init_manifest(
    manifest_path: Path,
    *,
    movie_id: str,
    selected_shots: List[ShotSpec],
    seed_base: int,
    warm_worker: bool,
) -> Dict[str, Any]:
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_by_num = {
            int(item["shot_number"]): item for item in manifest.get("shots", [])
        }
    else:
        manifest = {
            "movie_id": movie_id,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "sequence_total_shots_defined": len(TEST_SEQUENCE_SHOTS),
            "sequence_target_duration_seconds": sum(
                s.duration for s in TEST_SEQUENCE_SHOTS
            ),
            "shots": [],
            "assembled_movie": None,
        }
        existing_by_num = {}

    manifest["movie_id"] = movie_id
    manifest["warm_worker"] = warm_worker
    manifest["config_snapshot"] = {
        "video_provider": settings.VIDEO_PROVIDER,
        "width": settings.COGVIDEOX_WIDTH,
        "height": settings.COGVIDEOX_HEIGHT,
        "frames": settings.COGVIDEOX_FRAMES,
        "fps": settings.COGVIDEOX_FPS,
        "steps": settings.COGVIDEOX_STEPS,
        "guidance_scale": settings.COGVIDEOX_GUIDANCE_SCALE,
        "scheduler": settings.COGVIDEOX_SCHEDULER,
        "cpu_offload": settings.COGVIDEOX_CPU_OFFLOAD,
        "dtype": settings.COGVIDEOX_DTYPE,
    }

    for shot in selected_shots:
        if shot.shot_number not in existing_by_num:
            existing_by_num[shot.shot_number] = _initial_shot_entry(
                shot, seed_base + shot.shot_number
            )
        else:
            # Refresh immutable spec metadata while preserving checkpoint state.
            entry = existing_by_num[shot.shot_number]
            entry.update(
                {
                    "scene_description": shot.scene_description,
                    "camera_direction": shot.camera_direction,
                    "visual_prompt": shot.visual_prompt,
                    "duration": shot.duration,
                    "transition_information": shot.transition_information,
                }
            )

    manifest["shots"] = [
        existing_by_num[num] for num in sorted(existing_by_num)
    ]
    return manifest


def _is_completed_shot_valid(entry: Dict[str, Any]) -> bool:
    if entry.get("generation_status") != "completed":
        return False
    output_path_str = entry.get("output_path")
    if not output_path_str:
        return False
    output_path = Path(output_path_str)
    try:
        validation = validate_mp4(
            output_path,
            expected_width=settings.COGVIDEOX_WIDTH,
            expected_height=settings.COGVIDEOX_HEIGHT,
            expected_fps=float(settings.COGVIDEOX_FPS),
            expected_frames=settings.COGVIDEOX_FRAMES,
        )
    except Exception as exc:
        entry["generation_status"] = "failed"
        entry["validation_result"] = {"valid": False, "error": str(exc)}
        entry["error"] = f"Existing output failed validation on resume: {exc}"
        return False

    entry["validation_result"] = validation
    return True


async def run_batch(
    *,
    start_shot: int,
    end_shot: int,
    movie_id: str,
    seed_base: int,
    transition_duration: float,
    manifest_path: Path,
    max_retries: int,
    retry_shot: Optional[int],
    warm_worker: bool,
    assemble: bool,
    overwrite_movie: bool,
) -> Dict[str, Any]:
    selected_shots = [
        shot
        for shot in TEST_SEQUENCE_SHOTS
        if start_shot <= shot.shot_number <= end_shot
    ]
    if not selected_shots:
        raise ValueError(
            f"No shots matched range {start_shot}..{end_shot} (1..{len(TEST_SEQUENCE_SHOTS)})."
        )

    manifest = _load_or_init_manifest(
        manifest_path,
        movie_id=movie_id,
        selected_shots=selected_shots,
        seed_base=seed_base,
        warm_worker=warm_worker,
    )
    entries_by_num = {
        int(item["shot_number"]): item for item in manifest["shots"]
    }

    reused_shots: List[int] = []
    pending_shots: List[ShotSpec] = []
    for shot in selected_shots:
        entry = entries_by_num[shot.shot_number]
        if retry_shot is not None and shot.shot_number == retry_shot:
            entry["generation_status"] = "pending"
            entry["error"] = None
            pending_shots.append(shot)
        elif _is_completed_shot_valid(entry):
            reused_shots.append(shot.shot_number)
        else:
            if entry.get("generation_status") == "running":
                entry["error"] = "Previous run was interrupted before completion."
            pending_shots.append(shot)

    _save_manifest_atomic(manifest_path, manifest)

    video_service = VideoService()
    if video_service.video_provider.name != "cogvideox":
        raise RuntimeError(
            f"Expected active provider 'cogvideox', found '{video_service.video_provider.name}'."
        )

    newly_generated_shots: List[int] = []
    failed_shots: List[int] = []

    session_cm = (
        video_service.video_provider.warm_session()
        if warm_worker and isinstance(video_service.video_provider, CogVideoXVideoProvider)
        else contextlib.nullcontext()
    )

    if pending_shots:
        async with session_cm:
            for shot in pending_shots:
                entry = entries_by_num[shot.shot_number]
                total_allowed_attempts = 1 + max(0, max_retries)
                shot_succeeded = False

                for attempt_idx in range(total_allowed_attempts):
                    entry["attempts"] = int(entry.get("attempts", 0)) + 1
                    entry["generation_status"] = "running"
                    entry["error"] = None
                    _save_manifest_atomic(manifest_path, manifest)

                    request = VideoGenerationRequest(
                        scene_id=f"{movie_id}:shot-{shot.shot_number}",
                        story_id=movie_id,
                        prompt=shot.visual_prompt,
                        seed=int(entry["seed"]),
                        metadata={
                            "shot_number": shot.shot_number,
                            "scene_description": shot.scene_description,
                            "camera_direction": shot.camera_direction,
                            "transition_information": shot.transition_information,
                            "attempt": entry["attempts"],
                        },
                    )

                    started = time.perf_counter()
                    result = await video_service.generate_scene_video(request)
                    wall_seconds = time.perf_counter() - started

                    if result.status != "generated" or result.output_path is None:
                        entry["generation_status"] = "failed"
                        entry["generation_time"] = {
                            "wall_time_seconds": round(wall_seconds, 3),
                            "inference_time_seconds": None,
                        }
                        entry["validation_result"] = {
                            "valid": False,
                            "error": result.error or result.status,
                        }
                        entry["error"] = result.error or result.status
                        _save_manifest_atomic(manifest_path, manifest)
                        if attempt_idx + 1 < total_allowed_attempts:
                            continue
                        break

                    try:
                        validation = validate_mp4(
                            result.output_path,
                            expected_width=settings.COGVIDEOX_WIDTH,
                            expected_height=settings.COGVIDEOX_HEIGHT,
                            expected_fps=float(settings.COGVIDEOX_FPS),
                            expected_frames=settings.COGVIDEOX_FRAMES,
                        )
                    except Exception as exc:
                        # Remove invalid output file without touching any prior valid shot MP4.
                        with contextlib.suppress(OSError):
                            result.output_path.unlink(missing_ok=True)
                        entry["generation_status"] = "failed"
                        entry["generation_time"] = {
                            "wall_time_seconds": round(wall_seconds, 3),
                            "inference_time_seconds": round(
                                float(result.inference_time_seconds or 0.0), 3
                            ),
                        }
                        entry["validation_result"] = {
                            "valid": False,
                            "error": str(exc),
                        }
                        entry["error"] = f"MP4 validation failed: {exc}"
                        _save_manifest_atomic(manifest_path, manifest)
                        if attempt_idx + 1 < total_allowed_attempts:
                            continue
                        break

                    relative_video = result.output_path.resolve().relative_to(
                        settings.MEDIA_ROOT.resolve()
                    )
                    entry["output_path"] = str(result.output_path.resolve())
                    entry["video_url"] = f"/generated/{relative_video.as_posix()}"
                    entry["generation_status"] = "completed"
                    entry["generation_time"] = {
                        "wall_time_seconds": round(wall_seconds, 3),
                        "inference_time_seconds": round(
                            float(result.inference_time_seconds or 0.0), 3
                        ),
                    }
                    entry["validation_result"] = validation
                    entry["gpu_peak_allocated_mib"] = round(
                        result.metadata["gpu_peak_allocated_bytes"] / (1024**2), 1
                    )
                    entry["gpu_peak_reserved_mib"] = round(
                        result.metadata["gpu_peak_reserved_bytes"] / (1024**2), 1
                    )
                    entry["error"] = None
                    _save_manifest_atomic(manifest_path, manifest)
                    shot_succeeded = True
                    newly_generated_shots.append(shot.shot_number)
                    break

                if not shot_succeeded:
                    failed_shots.append(shot.shot_number)

    selected_entries = [entries_by_num[shot.shot_number] for shot in selected_shots]
    assembled_info: Optional[Dict[str, Any]] = None
    assembly_seconds: Optional[float] = None

    if assemble:
        if failed_shots or any(
            e.get("generation_status") != "completed" for e in selected_entries
        ):
            raise RuntimeError(
                f"Cannot assemble movie {movie_id}: incomplete or failed shots {failed_shots}. "
                f"Re-run with the same --movie-id / --manifest-path to resume."
            )

        assembly_scenes = [
            MovieSceneAssets(
                scene_number=int(entry["shot_number"]),
                duration_seconds=float(entry["validation_result"]["duration_seconds"]),
                video_path=entry["video_url"],
                transition_to_next=entry["transition_information"],
            )
            for entry in selected_entries
        ]
        output_movie_path = settings.MOVIES_DIR / f"{movie_id}.mp4"
        if overwrite_movie and output_movie_path.exists():
            output_movie_path.unlink()

        assembler = MovieAssembler()
        assembly_request = MovieAssemblyRequest(
            movie_id=movie_id,
            story_id=movie_id,
            title=f"Anant-Inspired Batch ({len(selected_shots)} shots)",
            scenes=assembly_scenes,
            fps=int(round(settings.COGVIDEOX_FPS)),
            resolution=f"{settings.COGVIDEOX_WIDTH}x{settings.COGVIDEOX_HEIGHT}",
            transition_duration_seconds=transition_duration,
            output_path=output_movie_path,
        )

        assembly_started = time.perf_counter()
        assembly_result = assembler.assemble(assembly_request)
        assembly_seconds = round(time.perf_counter() - assembly_started, 3)

        assembled_validation = validate_mp4(
            assembly_result.output_path,
            expected_width=assembly_result.width,
            expected_height=assembly_result.height,
            expected_fps=float(assembly_result.fps),
        )
        relative_movie = assembly_result.output_path.resolve().relative_to(
            settings.MEDIA_ROOT.resolve()
        )
        assembled_info = {
            "movie_url": f"/generated/{relative_movie.as_posix()}",
            "output_path": str(assembly_result.output_path),
            "duration_seconds": assembly_result.duration_seconds,
            "width": assembly_result.width,
            "height": assembly_result.height,
            "fps": assembly_result.fps,
            "scene_count": assembly_result.scene_count,
            "assembly_time_seconds": assembly_seconds,
            "metadata": assembly_result.metadata,
            "validation": assembled_validation,
        }
        manifest["assembled_movie"] = assembled_info
        _save_manifest_atomic(manifest_path, manifest)

    total_wall = sum(
        float(e["generation_time"]["wall_time_seconds"] or 0.0)
        for e in selected_entries
    )
    total_infer = sum(
        float(e["generation_time"]["inference_time_seconds"] or 0.0)
        for e in selected_entries
    )
    return {
        "manifest_path": str(manifest_path),
        "movie_id": movie_id,
        "warm_worker": warm_worker,
        "batch_range": f"{start_shot}..{end_shot}",
        "shots_in_batch": len(selected_entries),
        "reused_shots": reused_shots,
        "newly_generated_shots": newly_generated_shots,
        "failed_shots": failed_shots,
        "total_ai_wall_seconds": round(total_wall, 3),
        "total_ai_inference_seconds": round(total_infer, 3),
        "assembled_movie": assembled_info,
        "shots": selected_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a controlled, resumable multi-shot CineAI batch."
    )
    parser.add_argument("--start-shot", type=int, default=1)
    parser.add_argument("--end-shot", type=int, default=3)
    parser.add_argument(
        "--movie-id",
        type=str,
        default=f"poc_anant_3shots_{int(time.time())}",
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--transition-duration", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-shot", type=int, default=None)
    parser.add_argument("--warm-worker", action="store_true")
    parser.add_argument("--no-assemble", action="store_true")
    parser.add_argument("--overwrite-movie", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Required to run batches larger than 5 shots.",
    )
    args = parser.parse_args()

    batch_count = args.end_shot - args.start_shot + 1
    if batch_count > 5 and not args.allow_large:
        raise SystemExit(
            f"Refusing to run {batch_count} shots without --allow-large (safety limit is 5 shots)."
        )

    manifest_path = (
        args.manifest_path
        if args.manifest_path is not None
        else settings.MOVIES_DIR / f"{args.movie_id}.manifest.json"
    )

    report = asyncio.run(
        run_batch(
            start_shot=args.start_shot,
            end_shot=args.end_shot,
            movie_id=args.movie_id,
            seed_base=args.seed_base,
            transition_duration=args.transition_duration,
            manifest_path=manifest_path,
            max_retries=args.max_retries,
            retry_shot=args.retry_shot,
            warm_worker=args.warm_worker,
            assemble=not args.no_assemble,
            overwrite_movie=args.overwrite_movie,
        )
    )
    rendered = json.dumps(report, indent=2)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
