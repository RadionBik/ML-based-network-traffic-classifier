import typing
import pathlib

import sh


""" provided here for the sake of reproducibility of own research """


class Device(typing.NamedTuple):
    mac: str
    name: str
    category: str


IOT_DEVICES = [
    Device('d0:52:a8:00:67:5e', 'Smart Things', 'hub'),
    Device('44:65:0d:56:cc:d3', 'Amazon Echo', 'hub'),

    Device('70:ee:50:18:34:43', 'Netatmo Welcome', 'camera'),
    Device('f4:f2:6d:93:51:f1', 'TP-Link Day Night Cloud camera', 'camera'),
    Device('00:16:6c:ab:6b:88', 'Samsung SmartCam', 'camera'),
    Device('30:8c:fb:2f:e4:b2', 'Dropcam', 'camera'),
    Device('00:62:6e:51:27:2e', 'Insteon (wired)', 'camera'),
    Device('e8:ab:fa:19:de:4f', 'Insteon (wireless)', 'camera'),
    Device('00:24:e4:11:18:a8', 'Withings Smart Baby Monitor', 'camera'),

    Device('ec:1a:59:79:f4:89', 'Belkin Wemo', 'trigger'),
    Device('ec:1a:59:83:28:11', 'Belkin Wemo Motion sensor', 'trigger'),
    Device('50:c7:bf:00:56:39', 'TP-Link Smart Plug', 'trigger'),
    Device('74:c6:3b:29:d7:1d', 'iHome', 'trigger'),

    Device('18:b4:30:25:be:e4', 'NEST Protect smoke alarm', 'environment'),
    Device('70:ee:50:03:b8:ac', 'Netatmo weather station', 'environment'),

    Device('00:24:e4:1b:6f:96', 'Withings Smart scale', 'healthcare'),
    Device('00:24:e4:20:28:c6', 'Withings Aura smart sleep sensor', 'healthcare'),
    Device('74:6a:89:00:2e:25', 'Blipcare Blood Pressure meter', 'healthcare'),

    Device('d0:73:d5:01:83:08', 'LiFX Smart Bulb', 'light_bulb'),

    Device('18:b7:9e:02:20:44', 'Triby Speaker', 'electronics'),
    Device('e0:76:d0:33:bb:85', 'PIX-STAR photo-frame', 'electronics'),
    Device('70:5a:0f:e4:9b:c0', 'HP Printer', 'electronics'),
]


TCPDUMP_BASE_FILTER = 'not arp and not icmp and not icmp6 and not broadcast and not multicast and not net 127.0.0.0/8'


def _merge_pcaps(pcaps_to_merge: list, to_file):
    exec = sh.Command('mergecap')
    exec('-w', to_file, '-Fpcap', *pcaps_to_merge)


def _split_by_devices(source_pcap):
    exec = sh.Command('/usr/sbin/tcpdump')
    target_dir = source_pcap.parent / 'separated_iot_devices'
    target_dir.mkdir(exist_ok=True)
    for device in IOT_DEVICES:
        target_file = target_dir / f'{device.category}_{device.name.lower().replace(" ", "_")}.pcap'
        filter_str = f"ether host {device.mac} and not (dst net 192.168.1.0/24 and src net 192.168.1.0/24) " \
                     f"and {TCPDUMP_BASE_FILTER}"
        exec(['-r', source_pcap, filter_str, '-w', target_file])


def _filter_non_iot_dump(source_pcap):
    target_file = source_pcap.parent / 'non_iot.pcap'
    filter_str = f"not (dst net 192.168.88.0/24 and src net 192.168.88.0/24) and {TCPDUMP_BASE_FILTER}"
    exec = sh.Command('/usr/sbin/tcpdump')
    exec(['-r', source_pcap, filter_str, '-w', target_file])


def main():
    dump_root_dir = pathlib.Path('/media/raid_store/pretrained_traffic')
    merged_pcap = dump_root_dir / 'total.pcap'
    # merge all .pcap files from https://iotanalytics.unsw.edu.au/iottraces
    pcaps = pathlib.Path(dump_root_dir / 'iot_downloads').glob('*.pcap')
    _merge_pcaps(pcaps, merged_pcap)
    _split_by_devices(merged_pcap)
    _filter_non_iot_dump(dump_root_dir / 'home.pcap')


if __name__ == '__main__':
    main()
