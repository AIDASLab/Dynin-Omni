import hashlib
import json
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ..smp import LMUDataRoot, dump, load
from .video_base import VideoBaseDataset


_FAIL_MSG = 'Failed to obtain answer via API.'


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_choice(text: str) -> str:
    if text is None:
        return ""
    # Match (A), A), A., Option A, or standalone A
    m = re.search(r"\(?\b([A-E])\b\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    text = text.strip()
    if text and text[0].upper() in "ABCDE":
        return text[0].upper()
    return ""


def _extract_options_from_question(question: str) -> Dict[str, str]:
    opts = {}
    if not question:
        return opts
    # Match "A. xxx", "A) xxx", "(A) xxx"
    for m in re.finditer(r"\(?\b([A-E])\b\)?[\\.)]?\s*([^\n]+)", question):
        letter = m.group(1).upper()
        text = m.group(2).strip()
        if letter and text:
            opts[letter] = text
    return opts


def _dataset_from_path(video_path: str) -> str:
    if "ActivityNet" in video_path:
        return "ActivityNetQA"
    if "NExTQA" in video_path or "NextQA" in video_path:
        return "NExTQA"
    return "Unknown"


def _load_jsonl_multi(path: str) -> List[dict]:
    data: List[dict] = []
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
                continue
            except Exception:
                # Fall back to parsing multiple JSON objects in one line
                idx = 0
                n = len(line)
                parsed_any = False
                while idx < n:
                    while idx < n and line[idx].isspace():
                        idx += 1
                    if idx >= n:
                        break
                    try:
                        obj, end = decoder.raw_decode(line, idx)
                        data.append(obj)
                        parsed_any = True
                        idx = end
                    except Exception:
                        # Skip garbage tail if we already parsed at least one object
                        if parsed_any:
                            break
                        # Otherwise, re-raise to surface truly broken lines
                        raise
    return data


class VideoJSONLMCQ(VideoBaseDataset):
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Dynin_Omni_VideoJSONL', jsonl_path=None, nframe=0, fps=-1, force_video=False):
        self.jsonl_path = jsonl_path or os.environ.get(
            "DYNIN_OMNI_VIDEO_JSONL",
            "datasets/video/vlmeval_sft_train_rebuild_dedup.jsonl",
        )
        self.force_video = bool(force_video)
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['Dynin_Omni_VideoJSONL']

    def prepare_dataset(self, dataset):
        # Load jsonl into a temporary TSV under LMUDataRoot for consistent IO.
        data = _load_jsonl_multi(self.jsonl_path)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        if 'index' not in df:
            df['index'] = np.arange(len(df))
        tmp_dir = os.path.join(LMUDataRoot(), 'files')
        os.makedirs(tmp_dir, exist_ok=True)
        tsv_path = os.path.join(tmp_dir, f'{dataset}.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        # root is unused because we store absolute video paths
        return dict(root='/', data_file=tsv_path)

    def _frame_dir(self, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        h = hashlib.md5(video_path.encode('utf-8')).hexdigest()[:8]
        frame_root = os.path.join(LMUDataRoot(), 'images', self.dataset_name, f"{base}_{h}")
        os.makedirs(frame_root, exist_ok=True)
        return frame_root

    def _frame_paths(self, video_path: str, nframe: int) -> List[str]:
        frame_root = self._frame_dir(video_path)
        return [os.path.join(frame_root, f"frame_{i:02d}.jpg") for i in range(nframe)]

    def _save_video_frames(self, video_path: str) -> List[str]:
        if self.nframe <= 0 and self.fps <= 0:
            raise ValueError('fps and nframe should be set at least one valid value')

        # Use cached frames if present
        nframe = self.nframe
        if nframe > 0:
            frame_paths = self._frame_paths(video_path, nframe)
            if all(os.path.exists(p) for p in frame_paths):
                return frame_paths
        else:
            frame_paths = None

        try:
            import decord  # type: ignore

            vr = decord.VideoReader(video_path)
            total = len(vr)
            if total <= 0:
                raise RuntimeError("empty video")

            if self.fps > 0:
                video_fps = vr.get_avg_fps()
                total_duration = total / video_fps
                required_frames = int(total_duration * self.fps)
                step = video_fps / self.fps
                indices = [int(i * step) for i in range(required_frames)]
                frame_paths = self._frame_paths(video_path, len(indices))
            else:
                step = total / (self.nframe + 1)
                indices = [int(i * step) for i in range(1, self.nframe + 1)]
                frame_paths = self._frame_paths(video_path, self.nframe)

            images = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        except Exception:
            try:
                import cv2  # type: ignore

                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    raise RuntimeError("empty video")
                if self.fps > 0:
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    total_duration = total / video_fps
                    required_frames = int(total_duration * self.fps)
                    step = video_fps / self.fps
                    indices = set(int(i * step) for i in range(required_frames))
                    frame_paths = self._frame_paths(video_path, len(indices))
                else:
                    step = total / (self.nframe + 1)
                    indices = set(int(i * step) for i in range(1, self.nframe + 1))
                    frame_paths = self._frame_paths(video_path, self.nframe)

                images = []
                i = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if i in indices:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        images.append(Image.fromarray(frame))
                    i += 1
                cap.release()
            except Exception as exc:
                raise RuntimeError(f"Failed to read video {video_path}: {exc}") from exc

        for im, pth in zip(images, frame_paths):
            if not os.path.exists(pth):
                im.save(pth)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question']
        video_path = line['video']

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frame_paths = self._save_video_frames(video_path)
            for p in frame_paths:
                message.append(dict(type='image', value=p))
        message.append(dict(type='text', value=question))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        data = load(eval_file)
        totals: Dict[str, int] = {}
        corrects: Dict[str, int] = {}

        for _, row in data.iterrows():
            pred = str(row.get('prediction', ''))
            question = str(row.get('question', ''))
            answer = str(row.get('answer', ''))
            video_path = str(row.get('video', ''))
            ds = _dataset_from_path(video_path)

            pred_opt = _extract_choice(pred)
            gt_opt = _extract_choice(answer)
            if not gt_opt:
                # try match answer text to option
                opts = _extract_options_from_question(question)
                for k, v in opts.items():
                    if _normalize_text(v) == _normalize_text(answer):
                        gt_opt = k
                        break

            hit = False
            if gt_opt and pred_opt:
                hit = (gt_opt == pred_opt)
            else:
                # yes/no questions
                ans_norm = _normalize_text(answer)
                pred_norm = _normalize_text(pred)
                if ans_norm in {"yes", "no"}:
                    hit = (pred_norm == ans_norm)
                else:
                    # fallback: match option text
                    opts = _extract_options_from_question(question)
                    if gt_opt in opts:
                        if _normalize_text(opts[gt_opt]) in _normalize_text(pred):
                            hit = True

            totals[ds] = totals.get(ds, 0) + 1
            corrects[ds] = corrects.get(ds, 0) + int(hit)

        # dump a simple score json
        score = {ds: {'success': corrects.get(ds, 0), 'overall': totals.get(ds, 0)} for ds in totals}
        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', '_score.json')
        dump(score, score_file)
        return score
