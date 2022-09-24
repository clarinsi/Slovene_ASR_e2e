# silence all tqdm progress bars
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from version import __version__
import arrow
from typing import Union, List, Dict, Optional, Any
from pydantic import BaseModel
from time import time
from glob import glob
import petname
import shutil
from pydub import AudioSegment
import yaml
import os

from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging
import contextlib


if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

_AUDIO_DURATION_SECONDS_LIMIT = 300
_AUDIO_FILE_SIZE_LIMIT = 44 + _AUDIO_DURATION_SECONDS_LIMIT*16000*2
_use_gpu_if_available = True
_model_tag = "unknown"

class ASRModel(BaseModel):
  class Config:
    arbitrary_types_allowed = True
  tag: str
  nemo: ModelPT
  platform: str
  active: int
  remap: Dict[str,str]

start_time: str = None
models: Dict[str, ASRModel] = {}
num_requests_processed: int = None



app = FastAPI(
  title='ASR API',
  version=__version__,
  contact={
      "name": "Vitasis Inc.",
      "url": "https://vitasis.si/",
      "email": "info@vitasis.si",
  }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class TranscribeResponse(BaseModel):
  result: str

translateResponseExamples = {
  "result: string": {
    "value": {
      "result": "ogrožanja varnosti ljudi oziroma njihovega življenja"
    }
  },
}


class Model(BaseModel):
  tag: str
  workers: Dict[str,Any]
  features: Optional[Dict[str,Any]]
  info: Optional[Dict[str,Any]]

class HealthCheckResponse(BaseModel):
  status: int
  start_time: Optional[str]
  models: Optional[List[Model]]
  num_requests_processed: Optional[int]

healthCheckResponseExamples = {
  "serving": {
    "value": {
      "status": 0,
      "start_time": arrow.utcnow().isoformat(),
      "models": [
        { "tag": "sl-SI:GEN:nemo-1.5", "workers": { "platform": "gpu", "active": 2 } },
      ]
    }
  },
  "failed state": {
    "value": {
      "status": 2,
    }
  },
}


@app.get(
  "/api/healthCheck",
  description="Retrieve service health info.",
  response_model=HealthCheckResponse,
  responses={ 200: { "description": "Success", "content": { "application/json": { "examples": healthCheckResponseExamples } } } }
)
def health_check():
  _SERVICE_UNAVAILABLE_ = -1
  _PASS_ = 0
  _WARN_ = 1
  _FAIL_ = 2

  status: HealthCheckResponse = {'status': _SERVICE_UNAVAILABLE_}
  if not models:
    status = {'status': _FAIL_}
  else:
    status = {'status': _PASS_}
    min_workers_available = 1 # min([ workers_info['available'] for workers_info in _response['workers_info'] ]) if len(_response['workers_info']) > 0 else 0
    if min_workers_available <= -1: # config['workers']['fail']
      status = {'status': _FAIL_}
    elif min_workers_available <= 0: # config['workers']['warn']:
      status = {'status': _WARN_}
    status['models'] = [ { "tag": models[model_tag].tag, "workers": { "platform": models[model_tag].platform, "active": models[model_tag].active } } for model_tag in models ]
    status['start_time'] = start_time
    status['num_requests_processed'] = num_requests_processed

  return status

@app.post(
  "/api/transcribe",
  description=f"Transcribe audio file. No file format checking is performed. Maximum audio duration is {_AUDIO_DURATION_SECONDS_LIMIT}s.\n\nInput: 16bit 16kHz mono WAV.\n\nOutput: Transcript.",
  response_model=TranscribeResponse,
  responses={ 200: { "description": "Success", "content": { "application/json": { "examples": translateResponseExamples } } } }
)
def transcribe_file(audio_file: UploadFile = File( ..., title="Audio file", description="WAV, 16bit, 16kHz, mono")):
  time0 = time()
  session_path = f"/tmp/asr/{petname.generate()}"
  while os.path.exists(session_path):
    session_path = f"/tmp/asr/{petname.generate()}"
  os.makedirs(session_path)
  audio_file.filename=f"{session_path}/{audio_file.filename}"
  try:
    with open(f"{audio_file.filename}", 'wb') as f:
      shutil.copyfileobj(audio_file.file, f)
  except:
    raise HTTPException(status_code=400, detail=f"Bad request.")
  finally:
    audio_file.file.close()

  audio_file_size = os.path.getsize(f"{audio_file.filename}")
  if audio_file_size > _AUDIO_FILE_SIZE_LIMIT:
    logging.warning(f'{audio_file.filename}, file size exceded {audio_file_size}b [max {_AUDIO_FILE_SIZE_LIMIT}b]')
    shutil.rmtree(session_path, ignore_errors=True)
    raise HTTPException(status_code=400, detail=f"Bad request.")

  audio = AudioSegment.from_file(f"{audio_file.filename}")
  if audio.duration_seconds > _AUDIO_DURATION_SECONDS_LIMIT:
    logging.warning(f'{audio_file.filename}, audio duration exceded {len(audio)}ms [max {_AUDIO_DURATION_SECONDS_LIMIT}s]')
    shutil.rmtree(session_path, ignore_errors=True)
    raise HTTPException(status_code=400, detail=f"Bad request.")

  if _use_gpu_if_available and torch.cuda.is_available():
      models[_model_tag].nemo = models[_model_tag].nemo.cuda()

  models[_model_tag].active += 1
  try:
    with autocast():
      with torch.no_grad():
        transcriptions = models[_model_tag].nemo.transcribe([audio_file.filename], batch_size=32)

  except RuntimeError:
    logging.warning("Ran out of memory on device, performing inference on CPU for now")
    try:
      models[_model_tag].nemo = models[_model_tag].nemo.cpu()
      with torch.no_grad():
        transcriptions = models[_model_tag].nemo.transcribe([audio_file.filename], batch_size=32)

    except Exception as e:
      models[_model_tag].active -= 1
      logging.error(f"Exception {e} occured while attemting to transcribe audio. Returning error message")
      raise HTTPException(status_code=500, detail=f"Exception {e} occured while attemting to transcribe audio. Returning error message")

  # If RNNT models transcribe, they return a tuple (greedy, beam_scores)
  if type(transcriptions[0]) == list and len(transcriptions) == 2:
    # get greedy transcriptions only
    transcriptions = transcriptions[0]
  logging.debug(f' T: {transcriptions}')

  # Remap special chars
  for k,v in models[_model_tag].remap.items():
    for i in range(len(transcriptions)):
      transcriptions[i] = transcriptions[i].replace(k,v)
  logging.debug(f' T: {transcriptions}')

  models[_model_tag].active -= 1

  result: TranscribeResponse = { "result": transcriptions[0] }

  # cleanup
  shutil.rmtree(session_path, ignore_errors=True)

  transcription_duration = time()-time0
  logging.info(f' R: {audio_file.filename}, {result}')
  logging.debug(f'audio_duration: {round(len(audio)/1000,2)}s, transcription_duration: {round(transcription_duration,2)}s, RT: {round(len(audio)/(transcription_duration*1000),2)}x')
  global num_requests_processed
  num_requests_processed += 1

  if num_requests_processed == 0:
    if _use_gpu_if_available and torch.cuda.is_available():
      # Force onto CPU
      models[_model_tag].nemo = models[_model_tag].nemo.cpu()
      torch.cuda.empty_cache()

  return result


def initialize():
  time0 = time()
  models: Dict[str, ASRModel] = {}
  for _model_info_path in glob(f"./models/**/model.info",recursive=True):
    with open(_model_info_path) as f:
      _model_info = yaml.safe_load(f)

    global _model_tag
    _model_tag = f"{_model_info['language_code']}:{_model_info['domain']}:{_model_info['version']}"
    _model_platform = "gpu" if _use_gpu_if_available and torch.cuda.is_available() else "cpu"
    am=f"{_model_info['info']['am']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['am']['framework'].partition(':')[0]}"
    _model_path=os.path.join(os.path.dirname(_model_info_path),am)

    model = ModelPT.restore_from(_model_path,map_location="cuda" if _model_platform == "gpu" else "cpu")
    model.freeze()
    model.eval()

    models[_model_tag] = ASRModel(
      tag = _model_tag,
      nemo = model,
      platform = _model_platform,
      active = 0,
      remap = _model_info.get('features',[]).get('remap',[])
    )
  logging.info(f'Loaded models {[ (models[model_tag].tag,models[model_tag].platform) for model_tag in models ]}')
  logging.info(f'Initialization finished in {round(time()-time0,2)}s')

  start_time = arrow.utcnow().isoformat()
  num_requests_processed = 0
  return start_time, models, num_requests_processed

def start_service():
  uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
  logging.setLevel(logging.DEBUG)
  start_time, models, num_requests_processed = initialize()
  start_service()
