
'''
Pixelating an AGN catalog

Adapted from catalog.py in gwcosmo
https://git.ligo.org/lscsoft/gwcosmo/-/blob/master/gwcosmo/prior/catalog.py?ref_type=heads
'''

import os
import glob
import pickle
import array
import numpy as np
import h5py
from darksirenpop.utils import ipix_from_ra_dec, get_cachedir
# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp
from multiprocessing import Pool
from collections import defaultdict

DEG2RAD = np.pi/180.0


def get_catalogfile(filename):
    if 'AGN_CATALOG_PATH' in os.environ.keys():
        file = os.path.join(os.environ['AGN_CATALOG_PATH'], filename)
    else:
        file = filename
    if os.path.exists(file):
        return file
    else:
        raise FileNotFoundError(f"Unable to locate {filename}. Make sure your $AGN_CATALOG_PATH is set")


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    try:
        self._check_running()
    except AttributeError:
        if self._state != mpp.RUN:
            raise ValueError("Pool not running")
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    try:
        result = mpp.IMapIterator(self)
    except TypeError:
        result = mpp.IMapIterator(self._cache)

    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap


##### NEEDED? TODO ######
def load_catalog(name):
    # Load raw catalog
    if name == 'QUAIA':
        cat = OldStyleQuaia()
    elif name == 'MOCK':
        cat = OldStyleMock()
    else:
        raise ValueError(f"Unable to recognize catalog {name}. Supported: QUAIA or MOCK.")
    return cat

def load_catalog_from_opts(opts):
    name = opts.catalog
    return load_catalog(name)

def load_catalog_from_path(name, catalog_path):
    if name == 'QUAIA':
        cat = OldStyleQuaia(catalog_file=catalog_path)
    elif name == 'MOCK':
        cat = OldStyleMock(catalog_file=catalog_path)
    else:
        raise ValueError(f"Unable to recognize catalog {name}. Supported: QUAIA or MOCK.")
    return cat

###########################


class GalaxyCatalog:
    """
    Interface for a generic galaxy catalog
    """

    colnames = {'redshift', 'redshift_error', 'ra', 'dec'}

    def __init__(self, data = None, name = 'Unknown Catalog'):
        self.data = data
        self.name = name
        # Cache for pixel index to array index lookup
        self.pixmap = {}

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.data.__setitem__(*args, **kwargs)

    def __len__(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)
        

    def build_pixel_index_file(self, nside, cachedir=None, nested=True):
        if cachedir is None:
            cachedir = get_cachedir()

        filepath = self._cachefile(nside, nested=nested, cachedir=cachedir)
        pixlists = pixelate(self, nside, nested=nested)
        with open(filepath,'wb') as idxfile:
            pickle.dump(pixlists, idxfile)


    def _cachefile(self, nside, nested=True, cachedir=None):
        if cachedir is None:
            cachedir = get_cachedir()

        return os.path.join(cachedir,
                           '_'.join([self.name, str(nside), str(nested), 'pixidx.pkl'] )
                           )
    

    def read_pixel_index_cache(self, nside, cachedir=None):
        cachefile = self._cachefile(nside, cachedir=cachedir)
        if os.path.exists(cachefile):
            try:
                self.pixmap[nside] = pickle.load(open(cachefile,'rb'))
                return True
            except:
                print(f"Warning, unable to open pixel cache file {cachefile}, possible corrupted file")
                return False
        else:
            return False
        

    def clean_cache(self, mtime, cachedir=None):
        """
        Remove any index cache files older than mtime
        """
        if cachedir is None:
            cachedir = get_cachedir()
        for f in glob.glob(cachedir+'/'+self.name+'_*'):
            try:
                if os.path.getmtime(f) < mtime:
                    print('Clearing cache...')
                    os.remove(f)
            except FileNotFoundError:
                # Here in case a parallel job has removed the file
                pass


    def select_pixel(self, nside, pixel_index, nested=True):
        """
        Keep only galaxies in the desired healpix pixel indices

        Note from Lucas:
        Calling pixelate() on the fly breaks the multithreading. 
        Workaround: Try to make sure the correct pixel index file has been put in cache before threading -> see compute_zprior.py
        """
        # Try to load an index file, and create one if not
        if not self.read_pixel_index_cache(nside):
            # Try to make the index file
            self.build_pixel_index_file(nside, nested=nested)
            # Check if it can be read
            if not self.read_pixel_index_cache(nside):
                # If not, build it here
                print('No cache file found, generating pixel index (may take some time)')
                self.pixmap[nside] = pixelate(self, nside, nested=nested)

        pixmap = self.pixmap[nside]
        idx = pixmap[pixel_index]
        new = GalaxyCatalog(data = self.data[idx], name = self.name+f'_nside{nside}_pixel{pixel_index}')
        return new


    def idx2pixdict(self, nside, idx, nested=True):
        ra, dec = self['ra'][idx], self['dec'][idx]
        ipix = ipix_from_ra_dec(nside, ra, dec, nest=nested)
        return (ipix, idx)


    # def apply_redshift_cut(self, zcut):
    #     idx = np.where(((self['z']-3*self['sigmaz']) <= zcut))
    #     return GalaxyCatalog(data = self.data[idx], name = self.name+f'_zcut{zcut}')


def newarr():
    return array.array('Q')


def pixelate(cat, nside, allowed_pixels=None, nested=True):
    Nprocs = 2 # higher values use more RAM and disk bandwidth.

    # Arrays of unsigned 64-bit integers for the galaxy indices
    pixlists = defaultdict(newarr)

    with Pool(processes=Nprocs) as pool:

        for (pixidx, galidx) in \
            pool.istarmap(cat.idx2pixdict,
                          ( [nside, idx] for idx in range(len(cat)) ),
                          chunksize=1000000
                         ):
                pixlists[pixidx].append(galidx)
    for k, v in pixlists.items():
        pixlists[k] = np.sort(v)
    return pixlists


class OldStyleCatalog(GalaxyCatalog):
    """
    Catalog in the old GWCosmo format. Must have been
    preprocessed into HDF5 files.
    """
    filename = None
    def __init__(self,
                catalog_file=None,
                name = None):

        self.filename = get_catalogfile(catalog_file)  # TODO: just make a path I guess?

        super().__init__(name = name)
        self.populate()
        self.clean_cache(os.path.getmtime(self.filename))

    def populate(self):
        """
        This is a separate step to load the data
        """
        f = h5py.File(self.filename, 'r')
        names = []
        for n in self.colnames:
            if n in f:
                names.append(n)
            else:
                print(f'Unable to find column for {n}-band')
        self.data = np.rec.fromarrays([f[n] for n in names], names = names)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.populate()


# TODO: These are becoming obsolete, especially with the redshift array always constant - or make variable input
class OldStyleQuaia(OldStyleCatalog):
    def __init__(self, catalog_file = 'quaia.hdf5'):
        super().__init__(catalog_file = catalog_file, name = 'Quaia')

class OldStyleMock(OldStyleCatalog):
    def __init__(self, catalog_file):
        super().__init__(catalog_file, name = 'Mock')
