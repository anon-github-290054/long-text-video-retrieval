# long-text-video-retrieval

First, install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install pandas numpy decord
$ pip install git+https://github.com/openai/CLIP.git
```
### Method
query_scoring.py contains `similarity_queryscoring` function which is the proposed method.


### Zero-shot eval
Run query-scoring method with CLIP on MSR-VTT text-to-video retrieval.
N.B.: Downloads 6GB MSR-VTT videos to ./data

`python run_msr_retrieval --num_frames 16`

expected results for 16 frames:
```
{'R1': 34.5, 'R5': 56.1, 'R10': 65.3, 'R50': 85.5, 'MedR': 4.0, 'MeanR': 37.604} agg: query-scoring, temp: 0.1
{'R1': 33.4, 'R5': 55.0, 'R10': 65.1, 'R50': 84.7, 'MedR': 4.0, 'MeanR': 38.393} agg: mean-pooling, temp:
```

As you can see, query scoring provides a zero-shot performance improvement. This scoring function can be input to the loss function during training to achieve a more substantial boost.

You can play around with different temperature values (tau), when temperature << 1, it approximates max, when temperature >> 1 it approximates mean-pooling.

for best results, use 120 frames (takes longer, and needs more RAM, chunking required)
`python run_msr_retrieval --num_frames 120 --batch_size 1 --num_workers 1`






