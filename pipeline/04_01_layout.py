from typing import Optional, Literal
import os
from pathlib import Path
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy
import sys

print('TEST', file=sys.stderr)


class LayoutArgs(ClusterJobBaseArguments):
    model: Literal['minilm', 'bertopic'] = 'minilm'  # The embedding model to use.
    file_in: Optional[str] = None  # The file containing tweet embeddings (relative to source root)
    file_out: Optional[str] = None  # The file to write layout to (relative to source root)

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    projection: Literal['umap', 'tsne'] = 'tsne'  # The dimensionality reduction method to use

    # tsne args
    tsne_perplexity: int = 30
    tsne_exaggeration: float = None
    tsne_early_exaggeration_iter: int = 250
    tsne_early_exaggeration: float = 12
    tsne_initialization: Literal['random', 'pca', 'spectral'] = 'pca'
    tsne_metric: str = 'cosine'
    tsne_n_jobs: int = 8
    tsne_dof: float = 1.
    tsne_random_state: int = 3
    tsne_verbose: bool = True

    # umap args
    umap_n_neighbors: int = 15
    umap_n_components: int = 2
    umap_metric: str = 'cosine'
    umap_output_metric: str = 'euclidean'
    umap_min_dist: float = 0.1
    umap_spread: float = 1.0
    umap_local_connectivity: int = 1
    umap_repulsion_strength: float = 1.0
    umap_negative_sample_rate: int = 5
    umap_random_state: bool = None
    umap_densmap: bool = False
    umap_set_op_mix_ratio: float = 1.0
    umap_dens_lambda: float = 2.0
    umap_dens_frac: float = 0.3
    umap_dens_var_shift: float = 0.1

    cluster_jobname: str = 'twitter-layout'
    cluster_workdir: str = 'twitter'


if __name__ == '__main__':
    print('I will load pre-computed embeddings and reduce them to a 2D space!')
    args = LayoutArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = LayoutArgs().load(args.args_file)
    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_embeddings_{args.limit}_{_include_hashtags}_{args.model}.npy'
    else:
        file_in = args.file_in
    if args.file_out is None:
        file_out = f'data/{args.dataset}/topics/layout_{args.limit}_{args.projection}.npy'
    else:
        file_out = args.file_out

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)

        cluster_args = deepcopy(args)
        cluster_args.mode = 'local'
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        cluster_args.file_out = os.path.join(s_config.datadir_path, file_out)
        s_job.submit_job(main_script='pipeline/04_01_layout.py', params=cluster_args)
    else:
        import numpy as np

        print('Loading embeddings...')
        embeddings = np.load(file_in)

        if args.projection == 'umap':
            print('Fitting UMAP...')
            from umap import UMAP

            umap_args = {k[5:]: v for k, v in args.as_dict().items() if k.startswith('umap')}
            mapper = UMAP(**umap_args)
            layout = mapper.fit_transform(embeddings)
        elif args.projection == 'tsne':
            print('Fitting tSNE...')
            import openTSNE

            tsne_args = {k[5:]: v for k, v in args.as_dict().items() if k.startswith('tsne')}
            mapper = openTSNE.TSNE(**tsne_args)
            layout = mapper.fit(embeddings)
        else:
            raise NotImplementedError('Unknown projection method.')

        print('Storing layout...')
        os.makedirs(Path(file_out).parent, exist_ok=True)  # ensure the target directory exists first...
        np.save(file_out, layout)
