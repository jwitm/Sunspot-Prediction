"""Query JSOC and build the HDF5 archive used by the sunspot study.

The default command-line configuration follows the paper's observation
selection: definitive ``hmi.sharp_cea_720s`` records from 2013-11-14 through
2023-04-18 whose flux-weighted patch centers have Stonyhurst longitudes
strictly between -70 and 70 degrees. The selected HMI observables and auxiliary
masks are downloaded from JSOC and stored in one HDF5 group per HARP.

Warning:
    The complete dataset is approximately 7 TB according to the manuscript.
    Running this module only displays a query overview unless ``--download`` is
    explicitly supplied.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt_obj
from pathlib import Path

import drms
import h5py
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm


JSOC_BASE_URL = 'https://jsoc.stanford.edu'
PAPER_SERIES = 'hmi.sharp_cea_720s'
PAPER_TIME_RANGE = (
    '2013.11.14_00:00:00_TAI-2023.04.18_23:59:59_TAI'
)
PAPER_LONGITUDE_FILTER = '? LON_FWT > -70 AND LON_FWT < 70 ?'
PAPER_KEYWORDS = 'T_OBS, HARPNUM, NOAA_AR, LON_FWT'
PAPER_SEGMENTS = (
    'continuum, magnetogram, bitmap, Dopplergram, '
    'Bp, Bt, Br, conf_disambig'
)


class data_aq:
    """Query and download HMI SHARP records from JSOC.

    The class name is retained for compatibility with the original research
    scripts. For an unrestricted HARP query, set ``Harp_NUM=None`` and provide a
    DRMS record filter. A two-element HARP sequence is interpreted as an inclusive
    integer range.
    """

    def __init__(
        self,
        series,
        keywords,
        seg,
        Harp_NUM=None,
        time=None,
        filter=None,
        missing_harps_path='No_T_OBS.txt',
    ):
        """Query JSOC metadata for the requested records.

        Args:
            series: JSOC data-series name.
            keywords: Comma-separated metadata keywords requested from JSOC.
            seg: Comma-separated data-segment names.
            Harp_NUM: One HARP number, an inclusive ``(first, last)`` range, or
                ``None`` to query all HARPs matching ``filter``.
            time: DRMS time selector. Defaults to the unrestricted ``0-3000``.
            filter: Optional DRMS record filter, including surrounding question
                marks, for example ``"? LON_FWT > -70 ?"``.
            missing_harps_path: File in which HARP numbers without observations
                are recorded when a range is queried.

        Raises:
            ValueError: If neither a HARP selection nor a record filter is given.
        """
        super().__init__()
        self.time = time if time is not None else '0-3000'
        self.series = series
        self.keywords = keywords
        self.seg = seg
        self.filter = filter
        self.named_seg = [name.strip() for name in seg.split(',')]
        self.Dat_len = 0
        self.c = drms.Client()
        self.query = None

        if isinstance(Harp_NUM, int) or Harp_NUM is None:
            if Harp_NUM is None and filter is None:
                raise ValueError(
                    'Provide a HARP number/range or a DRMS record filter.'
                )
            self.Harp_NUM = [Harp_NUM] if Harp_NUM is not None else None
            self.query = self._build_query(Harp_NUM)
            self.keys = self.c.query(
                self.query,
                key=self.keywords,
            )
            if 'T_OBS' in self.keys and not self.keys.empty:
                print(
                    f'Download data in range '
                    f'{self.keys.T_OBS.min()}-{self.keys.T_OBS.max()}'
                )
                self.Dat_len = self.keys.T_OBS.size
            else:
                print("Keyword 'T_OBS' is absent or the query returned no records")
        else:
            self._query_harp_range(Harp_NUM, missing_harps_path)

    def _build_query(self, harp_number=None):
        """Construct a DRMS record-set query for one or all HARPs."""
        harp_clause = '' if harp_number is None else str(harp_number)
        query = f'{self.series}[{harp_clause}][{self.time}]'
        if self.filter is not None:
            query += f'[{self.filter}]'
        return query

    def _query_harp_range(self, harp_range, missing_harps_path):
        """Query every HARP in an inclusive numeric range."""
        if len(harp_range) != 2:
            raise ValueError('A HARP range must contain exactly two values')

        self.keys = []
        missing = []
        harp_numbers = range(int(harp_range[0]), int(harp_range[1]) + 1)
        self.Harp_NUM = []

        for number in tqdm(harp_numbers, desc='Querying HARPs'):
            query = self._build_query(number)
            keys = self.c.query(query, key=self.keywords)
            if 'T_OBS' in keys and not keys.empty:
                self.Dat_len += keys.T_OBS.size
                self.Harp_NUM.append(number)
                self.keys.append(keys)
            else:
                missing.append(number)

        print(f'{len(self.Harp_NUM)} HARPs to download')
        print(f'{len(missing)} HARPs to dismiss')
        missing_path = Path(missing_harps_path).expanduser()
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text(
            ''.join(f'{number}\n' for number in missing),
            encoding='utf-8',
        )

    @staticmethod
    def parse_tai_string(tstr, datetime=True):
        """Convert an HMI ``T_OBS`` string to a minute-resolution datetime.

        Args:
            tstr: HMI time string in ``YYYY.MM.DD_HH:MM:SS_TAI`` form.
            datetime: Return a ``datetime`` object when true; otherwise return
                ``(year, month, day, hour, minute)``.
        """
        year = int(tstr[:4])
        month = int(tstr[5:7])
        day = int(tstr[8:10])
        hour = int(tstr[11:13])
        minute = int(tstr[14:16])
        if datetime:
            return dt_obj(year, month, day, hour, minute)
        return year, month, day, hour, minute

    def overview(self, N=None):
        """Print approximate record count, storage, and download-time estimates.

        ``N`` is retained for compatibility and is currently unused. Estimates
        reproduce the rates recorded in the original script.
        """
        _ = N
        seconds_per_file = (16 * 60 + 13.9) / 2622
        gigabytes_per_file = 15 / 2622
        print(f'Total number of FITS files: {self.Dat_len}')
        print(
            f'Expected memory consumption: '
            f'{round(self.Dat_len * gigabytes_per_file, 1)} GB'
        )
        print(
            f'Expected download time: '
            f'{round(self.Dat_len * seconds_per_file / 3600, 1)} h'
        )
        if self.query is not None:
            print(f'DRMS query: {self.query}')

    @staticmethod
    def _read_fits(url):
        """Download one FITS segment and return an independent NumPy array."""
        with fits.open(url, memmap=False) as hdul:
            return np.asarray(hdul[-1].data, dtype=np.float32).copy()

    def download(self, filename='Data.h5', max_workers=None):
        """Download queried segments into a portable HDF5 archive.

        Args:
            filename: Destination HDF5 path. Parent directories are created.
            max_workers: Maximum concurrent FITS downloads. ``None`` uses the
                standard ``ThreadPoolExecutor`` default.

        Notes:
            Network reads occur concurrently, but HDF5 writes are performed in
            the main thread to avoid concurrent access to one HDF5 handle.
        """
        output_path = Path(filename).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        harp_numbers = self.Harp_NUM
        if harp_numbers is None:
            harp_numbers = np.unique(self.keys.HARPNUM).astype(int).tolist()

        with h5py.File(output_path, 'w') as data:
            progress = tqdm(harp_numbers, desc='Downloading HARPs')
            for harp_number in progress:
                query = self._build_query(harp_number)
                keys, segments = self.c.query(
                    query,
                    key=self.keywords,
                    seg=self.seg,
                )
                if 'T_OBS' not in keys or keys.empty:
                    continue

                harp_group = data.create_group(f'H_{harp_number}')
                observations = [
                    self.parse_tai_string(value, datetime=True).isoformat()
                    for value in keys.T_OBS
                ]
                harp_group.create_dataset(
                    't_obs',
                    data=np.asarray(observations, dtype='S23'),
                    maxshape=(None,),
                    chunks=True,
                )

                datasets = {}
                for segment_name in self.named_seg:
                    first_url = (
                        JSOC_BASE_URL + getattr(segments, segment_name).iloc[0]
                    )
                    first_frame = self._read_fits(first_url)
                    dataset = harp_group.create_dataset(
                        segment_name,
                        shape=(len(keys), *first_frame.shape),
                        dtype='f4',
                        chunks=True,
                    )
                    dataset[0] = first_frame
                    datasets[segment_name] = dataset

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for index in range(1, len(keys)):
                        for segment_name in self.named_seg:
                            url = (
                                JSOC_BASE_URL
                                + getattr(segments, segment_name).iloc[index]
                            )
                            future = executor.submit(self._read_fits, url)
                            futures[future] = (segment_name, index)

                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f'HARP {harp_number}',
                        leave=False,
                    ):
                        segment_name, index = futures[future]
                        datasets[segment_name][index] = future.result()

        print(f'Data saved to {output_path}')

    def get_data(self, filename='Data.h5'):
        """Load an HDF5 archive into nested dictionaries of torch tensors."""
        datasets = {}
        with h5py.File(Path(filename).expanduser(), 'r') as h5_file:
            for group_name in h5_file.keys():
                group = h5_file[group_name]
                datasets[group_name] = {'t_obs': group['t_obs'][:]}
                for dataset_name in self.named_seg:
                    datasets[group_name][dataset_name] = torch.from_numpy(
                        group[dataset_name][:]
                    )
        return datasets


def main():
    """Query the paper dataset and optionally download its HDF5 archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('Data_HMI_SHARP_20131114_20230418_lon70.h5'),
        help='Destination HDF5 file',
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Start the approximately 7 TB download after showing the overview',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum concurrent FITS downloads',
    )
    args = parser.parse_args()
    if args.max_workers is not None and args.max_workers < 1:
        parser.error('--max-workers must be positive')

    data = data_aq(
        PAPER_SERIES,
        PAPER_KEYWORDS,
        PAPER_SEGMENTS,
        Harp_NUM=None,
        time=PAPER_TIME_RANGE,
        filter=PAPER_LONGITUDE_FILTER,
    )
    data.overview()
    if args.download:
        data.download(args.output, max_workers=args.max_workers)
    else:
        print('Preview only. Pass --download to create the HDF5 archive.')


if __name__ == '__main__':
    main()
