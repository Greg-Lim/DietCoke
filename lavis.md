# set Up lavis

python 3.8 cant work
python 3.12 can work

python 3.10 works

pip install opencv-python
pip install imageio==1.1.3

### ModuleNotFoundError: No module named 'moviepy.editor'
pip install moviepy==1.0.3

### ImportError: cannot import name 'Cache' from 'transformers'
### ImportError: cannot import name 'DynamicCache' from 'transformers'
pip install transformers==4.47.1
pip install peft==0.10.0

### ImportError: cannot import name 'cached_download' from 'huggingface_hub'
https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2437615074
pip install huggingface_hub==0.25.0

### ImportError: cannot import name '_expand_mask' from 'transformers.models.clip.modeling_clip'
https://github.com/salesforce/LAVIS/issues/571
pip install transformers==4.31 - causes 'cache' error
4.33.0

### AssertionError: BLIP models are not compatible with transformers>=4.27, run pip install transformers==4.25 to downgrade


python -m spacy download en_core_web_sm

## Install lavis as a python module
```bash
pip install -e LAVIS
```

fastapi dev lavis_serve.py
