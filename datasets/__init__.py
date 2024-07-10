from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .ml_100k import ML100KDataset
from .ml_10m import ML10MDataset
from .Amazon_Office import AmazonOffice
from .Amazon_Toys import AmazonToys
from .Douban_Book import DoubanBookDataset
from .Douban_Movie import DoubanMovieDataset
from .Douban_Music import DoubanMusicDataset


DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    ML100KDataset.code():ML100KDataset,
    ML10MDataset.code():ML10MDataset,
    AmazonOffice.code():AmazonOffice,
    AmazonToys.code():AmazonToys,
    DoubanBookDataset.code():DoubanBookDataset,
    DoubanMovieDataset.code():DoubanMovieDataset,
    DoubanMusicDataset.code():DoubanMusicDataset,

}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
