import pathlib

import nfstream
import pandas as pd
import sh

import settings
from flow_parsing import parse_pcap_to_csv


def parse_flow_sizes(pcap_folder, target_folder):

    for pcap_file in pcap_folder.glob('*.pcap'):
        print(f'parsing {pcap_file}')
        dest_file = target_folder / (pcap_file.stem + '.csv')

        streamer = nfstream.NFStreamer(
            source=pcap_file.as_posix(),
            statistical_analysis=True,
            idle_timeout=settings.IDLE_TIMEOUT,
            active_timeout=settings.ACTIVE_TIMEOUT_ONLINE,
            accounting_mode=1,   # IP size,
        )
        print(f'saving to {dest_file}')
        streamer.to_csv(path=dest_file)


def parse_raw_features_from_pcaps(pcap_folder, target_folder):
    for pcap_file in pcap_folder.glob('*.pcap'):
        target_csv = target_folder / (pcap_file.stem + '.csv')
        if target_csv.exists():
            continue
        print(f'started parsing file {pcap_file}')

        # raw_features are set via analysis of packet number distribution within sessions
        # @ mawi.wide.ad.jp/mawi/ditl/ditl2020/ pcaps, such that the limit is close to .99 percentile

        parse_pcap_to_csv(pcap_file.as_posix(),
                          target_csv.as_posix(),
                          derivative_features=False,
                          raw_features=128,
                          provide_labels=True)


def record_session_lengths(target_folder):
    dfs = []
    for csv in target_folder.glob('*.csv'):
        df = pd.read_csv(csv, usecols=['bidirectional_packets'])
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    counts = dfs.bidirectional_packets.value_counts()
    norm_counts = counts.sort_index().cumsum() / dfs.shape[0]
    norm_counts.to_json(target_folder.parent / 'pkt_len_norm_counts.json')

    norm_counts_no_1packet_flows = (counts.sort_index().cumsum() - counts[1]) / (dfs.shape[0] - counts[1])
    norm_counts_no_1packet_flows.to_json(target_folder.parent / 'pkt_len_norm_counts_no_1_packet.json')


def rm_icmp_from_pcaps(source_pcap_folder, target_pcap_folder):
    for source_pcap in source_pcap_folder.glob('*.pcap'):
        target_pcap = target_pcap_folder / (source_pcap.stem + 'no_icmp.pcap')
        exec = sh.Command('/usr/sbin/tcpdump')
        exec(['-r', source_pcap, 'not icmp', '-w', target_pcap])


def split_pcaps_into_smaller(source_folder, dest_folder, size_limit=2000):
    for source_pcap in source_folder.glob('*.pcap'):
        target_pcaps = dest_folder / source_pcap.stem
        exec = sh.Command('/usr/sbin/tcpdump')
        exec(['-r', source_pcap, '-w', target_pcaps, '-C', size_limit])


def pcapng_to_pcap(pcap_folder):
    for source_pcap in pcap_folder.glob('*.pcapng'):
        target_pcap = pcap_folder / (source_pcap.stem + '.pcap')
        exec = sh.Command('tshark')
        exec(['-F', 'pcap', '-r', source_pcap, '-w', target_pcap])


def add_pcap_suffix(folder):
    for file in folder.glob('*'):
        file.replace(file.parent / (file.stem + '.pcap'))


def uncompress_and_split_pcaps(source_folder, target_folder):
    """
    bash script:

    for f in *.gz; do
      STEM_with_pcap=$(basename "${f}" .gz)
      STEM=$(basename "${STEM_with_pcap}" .pcap)
      # gunzip -c "${f}" > /media/raid_store/pretrained_traffic/mawi_pcaps/"${STEM}"
      gunzip -c "${f}" | tcpdump -w /media/raid_store/pretrained_traffic/mawi_pcaps/"${STEM}" -C 2000 -r -
    done

    :param folder:
    :return:
    """
    gunzip = sh.Command('gunzip')
    tcpdump = sh.Command('tcpdump')
    target_folder = pathlib.Path(target_folder)
    for file in source_folder.glob('*.gz'):
        stem = file.stem.split('.pcap')[0]
        target = target_folder / stem
        # not tested :) see https://amoffat.github.io/sh/sections/piping.html#piping
        tcpdump(gunzip('-c', file), '-w', target, '-C', 2000, '-r', '-')


if __name__ == '__main__':
    source_pcap_folder = pathlib.Path('/media/raid_store/pretrained_traffic/separated_iot_pcaps')

    # no_icmp_pcaps = pathlib.Path('/media/raid_store/pretrained_traffic/MAWI_no_icmp')
    # rm_icmp_from_pcaps(source_pcap_folder, no_icmp_pcaps)

    # clean_pcap_folder = pathlib.Path('/media/raid_store/pretrained_traffic/pcaps')
    # clean_pcap_folder.mkdir(exist_ok=True)
    # split_pcaps_into_smaller(clean_pcap_folder, split_pcap_folder, 2000)

    # split_pcap_folder = pathlib.Path('/media/raid_store/pretrained_traffic/ISCXVPN2016')
    # split_pcap_folder = pathlib.Path('/media/raid_store/pretrained_traffic/pcaps')
    # split_pcap_folder.mkdir(exist_ok=True)

    # add_pcap_suffix(source_pcap_folder)
    # parse_flow_sizes(split_pcap_folder, target_csv_folder_w_lengths)

    # target_csv_folder_w_lengths = pathlib.Path('/media/raid_store/pretrained_traffic/raw_csv_len')
    # target_csv_folder_w_lengths.mkdir(exist_ok=True)
    # record_session_lengths(target_csv_folder_w_lengths.parent)

    target_csv_folder = pathlib.Path('/media/raid_store/pretrained_traffic/raw_csv_iot_devices')
    target_csv_folder.mkdir(exist_ok=True)

    # pcapng_to_pcap(split_pcap_folder)

    parse_raw_features_from_pcaps(source_pcap_folder, target_csv_folder)
