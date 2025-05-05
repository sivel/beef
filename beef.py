#!/usr/bin/env python
# MIT License
#
# Copyright (c) Matt Martz <matt@sivel.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import contextlib
import ctypes
import functools
import json
import os
import pathlib
import pickle
import random
import re
import shutil
import socket
import subprocess
import sys
import termios
import textwrap
import time
import types
import typing as t
from dataclasses import asdict, dataclass, field, fields

__version__ = '0.0.1'


CLONE_NOFOLLOW = 1  # Don't follow symbolic links
CLONE_NOOWNERCOPY = 2  # Don't copy ownership information from source
CLONE_ACL = 4  # Copy access control lists from source
CLONE_NOFOLLOW_ANY = 8  # Don't follow any symbolic links in the path

_LIBC = ctypes.CDLL(None)
clonefile = _LIBC.clonefile
clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pathlib.Path):
            return str(obj)

        return super().default(obj)


def generate_laa_mac() -> str:
    first_byte = random.randint(0x00, 0xFF)
    first_byte = (first_byte & 0b11111100) | 0b00000010
    mac = [first_byte] + [random.randint(0x00, 0xFF) for _ in range(5)]
    return ':'.join(f'{b:02x}' for b in mac)


@dataclass(kw_only=True, slots=True)
class RunConfig:
    action: t.Callable[[t.Self], None] | None = field(
        default=None,
        metadata={'noarg': True}
    )
    vm: str = field(metadata={'help': 'Name of VM', 'posarg': True})
    src_image: pathlib.Path | None = field(
        default=None,
        metadata={
            'nargs': '?',
            'help': 'Path to raw cloud image',
            'posarg': True,
        },
    )
    storage: pathlib.Path = field(
        default=pathlib.Path.home() / 'vms' / 'storage',
        metadata={'help': 'Path to vmstorage dir'},
    )
    resize: str = field(
        default='+10',
        metadata={
            'help': (
                'Resize the disk in GB. Can be exact, or start with + or - '
                'to indicate a relative size change'
            )
        },
    )
    cpus: int = field(default=2, metadata={'help': 'Number of CPUs'})
    memory: int = field(
        default=2048,
        metadata={'help': 'Amount of memory in MB'},
    )
    user_data: pathlib.Path = field(
        default=pathlib.Path.home() / 'vms' / 'user-data',
        metadata={
            'name': '--user-data',
            'dest': 'user_data',
            'help': 'Path to cloud-init user_data file',
        },
    )
    volumes: list[tuple[pathlib.Path, str]] = field(
        default_factory=list,
        metadata={
            'name': '-v',
            'dest': 'volumes',
            'action': 'append',
            'type': str,
            'help': (
                'Volumes to mount into the VM. May be specified multiple '
                'times'
            ),
            'metavar': 'src:dst',
        }
    )
    mac: str = field(
        default=generate_laa_mac(),
        metadata={'help': 'MAC address'},
    )
    gui: bool = field(default=False, metadata={'help': 'Enable GUI'})
    attach: bool = field(
        default=False,
        metadata={
            'help': 'Attach and to VM consolel and run in foreground',
            'alias': '-a'
        },
    )
    force: bool = field(
        default=False,
        metadata={'help': 'Force rebuild of VM', 'alias': '-f'},
    )

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name, None)
            if value is not None and f.type is pathlib.Path:
                if not isinstance(value, pathlib.Path):
                    value = pathlib.Path(value)
                setattr(
                    self,
                    f.name,
                    value.resolve()
                )
            elif f.name == 'volumes':
                setattr(
                    self,
                    f.name,
                    self._parse_volumes(value),  # type: ignore[arg-type]
                )

    @t.overload
    def _parse_volumes(
            self,
            value: list[str]
    ) -> list[tuple[pathlib.Path, str]]:
        ...

    @t.overload
    def _parse_volumes(
            self,
            value: list[tuple[pathlib.Path, str]]
    ) -> list[tuple[pathlib.Path, str]]:
        ...

    def _parse_volumes(self, value):
        volumes = []
        for v in value:
            if isinstance(v, (list, tuple)):
                if isinstance(v[0], pathlib.Path):
                    volumes.append(tuple(v))
                else:
                    volumes.append((
                        pathlib.Path(v[0]).resolve(),
                        v[1]
                    ))
                continue
            src, dst = v.split(':', 1)
            volumes.append((
                pathlib.Path(src).resolve(),
                dst
            ))
        return volumes

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> t.Self:
        delattr(args, 'state_file')
        return cls(**vars(args))  # pylint: disable=missing-kwoa

    @property
    def vm_storage(self) -> pathlib.Path:
        return pathlib.Path(self.storage).joinpath(self.vm)

    @property
    def vm_disk(self) -> pathlib.Path:
        return pathlib.Path(
            self.storage
        ).joinpath(
            self.vm, self.vm
        ).with_suffix('.raw').resolve()

    @property
    def sock(self) -> pathlib.Path:
        return self.vm_disk.with_suffix('.sock')

    @property
    def pid(self) -> pathlib.Path:
        return self.vm_disk.with_suffix('.pid')

    @property
    def state_file(self) -> pathlib.Path:
        return self.storage / self.vm / 'run_config'

    def write(self) -> None:
        run_config = asdict(self)
        run_config.pop('force')
        run_config.pop('attach')
        run_config.pop('src_image')
        run_config.pop('action')
        state_file = self.state_file
        state_file.parent.mkdir(exist_ok=True)
        with (state_file).open('w') as f:
            json.dump(run_config, f, indent=4, cls=_JSONEncoder)

    @classmethod
    def read(cls, run_config: pathlib.Path) -> t.Self:
        with run_config.open('rb') as f:
            if f.read(1) == b'\x80':
                f.seek(0)
                return cls(**pickle.load(f))
            f.seek(0)
            return cls(**json.load(f))


def parse_args(argv: list[str] | None = None) -> RunConfig:
    name_parser = argparse.ArgumentParser(exit_on_error=False, add_help=False)
    name_subparsers = name_parser.add_subparsers(dest='action', required=True)

    name_subparsers.add_parser(  # type: ignore[attr-defined]
        'run', exit_on_error=False, add_help=False
    ).add_argument('vm').container.add_argument(
        'src_image', nargs='?'
    ).container.add_argument(
        '--state-file', type=pathlib.Path
    ).container.set_defaults(
        action=run
    )
    for func in (stop, status, rm):
        name_subparsers.add_parser(  # type: ignore[attr-defined]
            func.__name__, exit_on_error=False
        ).add_argument('vm').container.set_defaults(
            action=func
        )
    name_subparsers.add_parser(  # type: ignore[attr-defined]
        'ls', exit_on_error=False
    ).add_argument('--json', action='store_true').container.set_defaults(
        action=ls
    )

    try:
        name_args, _ = name_parser.parse_known_args(argv)
    except argparse.ArgumentError:
        name_args = argparse.Namespace()

    vm = getattr(name_args, 'vm', '')
    state_file = getattr(name_args, 'state_file', None)
    src_image = getattr(name_args, 'src_image', None)
    action = getattr(name_args, 'action', run)

    if state_file and not src_image:
        raise ValueError('src_image required')

    run_config = RunConfig(vm=vm, action=action)
    if action.__name__ == 'ls':
        run_config.action = functools.partial(  # type: ignore[call-arg]
            action,
            use_json=name_args.json
        )
        return run_config

    state_file = state_file or run_config.state_file
    if vm and state_file.exists():
        run_config = RunConfig.read(state_file)
        run_config.action = action

    if action.__name__ in ('stop', 'status', 'rm'):
        return run_config

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    run_parser = subparsers.add_parser('run')

    for name in ('stop', 'status', 'rm'):
        subparsers.add_parser(name).add_argument('vm', help='Name of VM')
    subparsers.add_parser('ls').add_argument('--json', action='store_true')

    for f in fields(RunConfig):
        kwargs = f.metadata.copy()
        if kwargs.get('noarg', False):
            continue

        name = kwargs.pop('name', None)
        alias = kwargs.pop('alias', None)
        posarg = kwargs.pop('posarg', False)
        default = getattr(run_config, f.name)
        if not posarg:
            kwargs['default'] = default

        if not (arg_type := kwargs.get('type')):
            if isinstance(f.type, types.UnionType):
                kwargs['type'] = arg_type = f.type.__args__[0]
            else:
                kwargs['type'] = arg_type = f.type

        if arg_type is bool:
            kwargs['action'] = 'store_true'
            kwargs.pop('type')

        if arg_type is not bool and default:
            kwargs['help'] += f'. Default: {default}'

        if name:
            flags = [name]
        elif posarg:
            flags = [f.name]
        elif len(f.name) == 1:
            flags = [f'-{f.name}']
        else:
            flags = [f'--{f.name}']

        if alias:
            flags.append(alias)

        run_parser.add_argument(*flags, **kwargs)
    run_parser.add_argument('--state-file', help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    if not state_file.exists() or vm:
        run_config = RunConfig.from_argparse(args)

    run_config.action = run

    vm_disk = run_config.vm_disk
    if vm_disk.exists() and not run_config.force and run_config.src_image:
        run_config.src_image = None

    if args.attach:
        run_config.attach = args.attach

    return run_config


def get_vm_ip(mac) -> str:
    mac = re.sub(r'(^|:)0', r'\1', mac)
    arp_re = re.compile(rf'^\S+ \(([^)]+)\) at {mac}', flags=re.M)
    for _ in range(30):
        try:
            p = subprocess.run(
                ['arp', '-an', '-i', 'bridge100'],
                check=True,
                capture_output=True,
                text=True,
            )
            ip_match = arp_re.search(p.stdout)
            if not ip_match:
                time.sleep(1)
                continue
            else:
                break
        except subprocess.SubprocessError:
            time.sleep(1)
            continue
    if ip_match:
        return ip_match.group(1)
    return 'UNKNOWN'


def resize(vm_disk, size):
    if size.isdigit():
        new = int(size) * 1024**3
    else:
        current = vm_disk.stat().st_size
        new = current + int(size) * 1024**3
    os.truncate(vm_disk, new)


@contextlib.contextmanager
def _make_sock(run_config: RunConfig):
    pid = run_config.pid
    if not pid.parent.is_dir():
        raise ValueError(f'No such VM: {run_config.vm}')

    if not pid.is_file():
        raise RuntimeError(
            f'{run_config.vm} is not running'
        )

    rest_sock = run_config.sock
    if not rest_sock.is_socket():
        raise RuntimeError('rest sock is missing')

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(rest_sock))
        yield sock


def rm(run_config: RunConfig):
    if run_config.pid.is_file():
        raise RuntimeError(
            f'{run_config.vm} is running'
        )
    shutil.rmtree(run_config.vm_disk.parent)


def ls(run_config: RunConfig, use_json=False) -> None:
    out = {}
    for vm_disk in run_config.storage.glob('*/*.raw'):
        vm = vm_disk.parent.name
        ip = True if use_json else False
        state = status(
            RunConfig.read(RunConfig(vm=vm).state_file),
            ip=ip,
            ret=True
        )
        if state:
            if not use_json:
                print(f'{vm}: {state["state"]}')
                continue
            out[vm] = state
    if use_json:
        print(json.dumps(out, indent=4))


def status(
        run_config: RunConfig,
        ip: bool = True,
        ret: bool = False
) -> dict[str, t.Any] | t.NoReturn:
    data: dict[str, t.Any]
    try:
        with _make_sock(run_config) as sock:
            sock.sendall(textwrap.dedent('''
                GET /vm/state HTTP/1.1
                Host: localhost

            ''').lstrip().encode())
            resp = sock.recv(1024).decode()
    except (RuntimeError, socket.error):
        rc = 2
        data = {"state": "Stopped"}
        if run_config.pid.is_file():
            run_config.pid.unlink(missing_ok=True)
            run_config.sock.unlink(missing_ok=True)
    else:
        rc = 0
        data = json.loads(resp.partition('\r\n\r\n')[2])
        if ip:
            data['ipAddress'] = get_vm_ip(run_config.mac)
        data['pid'] = int(run_config.pid.read_text())

    if ret:
        return data

    print(json.dumps(data, indent=4))
    sys.exit(rc)


def stop(run_config: RunConfig) -> None:
    with _make_sock(run_config) as sock:
        sock.sendall(textwrap.dedent('''
            POST /vm/state HTTP/1.1
            Host: localhost
            Content-Type: application/json
            Content-Length: 17

            {"state": "Stop"}
        ''').lstrip().encode())
        if sock.recv(21) != b'HTTP/1.1 202 Accepted':
            raise RuntimeError(f'Could not issue stop to {run_config.vm}')
    run_config.pid.unlink()
    run_config.sock.unlink()


def run(run_config: RunConfig) -> None:
    run_config.storage.mkdir(exist_ok=True)

    pid = run_config.pid
    if pid.is_file():
        status(run_config, ip=False, ret=True)
        if pid.is_file():
            raise RuntimeError(
                f'{run_config.vm}[{int(pid.read_text())}] is already running'
            )

    run_config.write()

    run_config.vm_storage.mkdir(exist_ok=True)
    if run_config.force and run_config.src_image:
        run_config.vm_disk.unlink(missing_ok=True)

    src_image = run_config.src_image
    if src_image:
        if run_config.vm_disk.is_file():
            raise ValueError(f'{run_config.vm_disk} already exists')

        if src_image.is_dir() and src_image.suffix == '.bundle':
            for file in ('AuxiliaryStorage', 'HardwareModel',
                         'MachineIdentifier'):
                shutil.copy2(src_image / file, run_config.vm_storage)

            src_image = src_image.joinpath(  # type: ignore[union-attr]
                'Disk.img'
            )

        rc = clonefile(
            bytes(src_image),
            bytes(run_config.vm_disk),
            CLONE_NOFOLLOW
        )
        if rc == -1:
            raise OSError(
                f'Could not clone {src_image} to '
                f'{run_config.vm_disk}'
            )

    is_mac = run_config.vm_storage.joinpath('AuxiliaryStorage').exists()

    if not run_config.vm_disk.is_file():
        raise ValueError(
            f'Could not locate disk image for {run_config.vm_disk.stem}'
        )

    if run_config.resize and (run_config.src_image or run_config.force):
        resize(run_config.vm_disk, run_config.resize)

    efi = run_config.vm_disk.with_suffix('.efi')
    if run_config.force:
        efi.unlink(missing_ok=True)
    rest_sock = run_config.sock
    if rest_sock.is_file():
        rest_sock.unlink()

    cmd = [
        'vfkit',
        '--cpus', f'{run_config.cpus}',
        '--memory', f'{run_config.memory}',
        '--device', f'virtio-blk,path={run_config.vm_disk}',
        '--device', f'virtio-net,nat,mac={run_config.mac}',
        '--device', 'virtio-rng',
        '--restful-uri', f'unix:{rest_sock}',
    ]

    if is_mac:
        vm_storage = run_config.vm_storage
        cmd.extend([
            '--bootloader',
            (
                'macos,'
                f'machineIdentifierPath={vm_storage / "MachineIdentifier"},'
                f'hardwareModelPath={vm_storage / "HardwareModel"},'
                f'auxImagePath={vm_storage / "AuxiliaryStorage"}'
            )
        ])
    else:
        cmd.extend([
            '--bootloader', f'efi,variable-store={efi},create',
        ])

    mounts = []
    for src, dst in run_config.volumes:
        if not src.exists():
            src.mkdir()
        tag = dst.replace(os.sep, '-')
        cmd.extend([
            '--device',
            f'virtio-fs,sharedDir={src},mountTag={tag}',
        ])
        mounts.append([
            tag, dst, 'virtiofs'
        ])

    user_data = run_config.vm_disk.parent / 'user-data'
    if not run_config.force and user_data.is_file():
        cmd.extend(['--cloud-init', f'{user_data}'])
    elif mounts or run_config.user_data.is_file():
        with user_data.open('w') as f:
            if run_config.user_data.is_file():
                f.write(run_config.user_data.read_text())
            else:
                f.write('#cloud-config\n')
            if mounts:
                f.write(f'\nmounts: {json.dumps(mounts)}\n')
            f.flush()
        cmd.extend(['--cloud-init', f'{user_data}'])

    if is_mac or run_config.gui:
        cmd.extend([
            '--device', 'virtio-input,keyboard',
            '--device', 'virtio-input,pointing',
            '--device', 'virtio-gpu,width=1024,height=800',
            '--gui',
        ])

    if run_config.attach:
        cmd.extend(['--device', 'virtio-serial,stdio'])

    popen_kwargs: dict[str, t.Any] = {}
    if not run_config.attach:
        popen_kwargs.update({
            'start_new_session': True,
            'stdin': subprocess.PIPE,
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
        })

    p = subprocess.Popen(cmd, **popen_kwargs)
    p.poll()

    run_config.pid.write_text(f'{p.pid}\n')

    for _ in range(3):
        if rest_sock.is_socket():
            break
        time.sleep(1)
    json.dump(
        status(run_config, ret=True),
        sys.stdout,
        indent=4,
    )

    if not run_config.attach:
        return

    try:
        p.wait()
    finally:
        rest_sock.unlink(missing_ok=True)
        pid.unlink(missing_ok=True)


@contextlib.contextmanager
def _repair_stdin():
    fd = sys.stdin.fileno()
    attrs = termios.tcgetattr(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)


def _verify_vfkit() -> None:
    if not shutil.which('vfkit'):
        print(
            (
                'vfkit not found, please install vfkit '
                '(https://github.com/crc-org/vfkit)'
            ),
            file=sys.stderr
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    _verify_vfkit()
    run_config = parse_args(argv)

    try:
        with _repair_stdin():
            run_config.action(run_config)  # type: ignore[misc]
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f'{e}', file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f'{e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
