#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
#
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
import collections.abc as c
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

try:
    import argcomplete
    HAS_ARGCOMPLETE = True
except ModuleNotFoundError:
    HAS_ARGCOMPLETE = False

__version__ = '0.0.1'


CLONE_NOFOLLOW = 1  # Don't follow symbolic links
CLONE_NOOWNERCOPY = 2  # Don't copy ownership information from source
CLONE_ACL = 4  # Copy access control lists from source
CLONE_NOFOLLOW_ANY = 8  # Don't follow any symbolic links in the path

_LIBC = ctypes.CDLL(None)
clonefile = _LIBC.clonefile
clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)

_DEFAULT_GUI_RESOLUTION = '1024x800'


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pathlib.Path):
            return str(obj)

        return super().default(obj)


def storage_completer(
        prefix: str,
        parsed_args: argparse.Namespace,
        **kwargs
) -> c.Iterator:
    storage = pathlib.Path(parsed_args.storage)
    return (p.name for p in storage.glob(f'{prefix}*') if p.is_dir())


def generate_laa_mac() -> str:
    first_byte = random.randint(0x00, 0xFF)
    first_byte = (first_byte & 0b11111100) | 0b00000010
    mac = [first_byte] + [random.randint(0x00, 0xFF) for _ in range(5)]
    return ':'.join(f'{b:02x}' for b in mac)


class Resize(str):
    def __new__(cls, value):
        if int(value) < 0:
            raise ValueError('negative values not supported')
        return str(value)


class Resolution(str):
    def __new__(cls, value):
        width, height = value.split('x')
        int(width)
        int(height)
        return str(value)


@dataclass(kw_only=True, slots=False)
class RunConfig:
    vm: str
    src_image: pathlib.Path | None = None
    storage: pathlib.Path
    resize: Resize | None = None
    cpus: int | None = None
    memory: int | None = None
    user_data: pathlib.Path | None = None
    volumes: list[tuple[pathlib.Path, str]] = field(
        default_factory=list,
    )
    mac: str | None = None
    gui: Resolution | None = None
    attach: bool = False
    force: bool = False

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name, None)
            if isinstance(f.type, types.UnionType):
                f_type = f.type.__args__[0]
            else:
                f_type = f.type
            if value is not None and f_type is pathlib.Path:
                if not isinstance(value, pathlib.Path):
                    value = pathlib.Path(value)
                setattr(
                    self,
                    f.name,
                    value.resolve()
                )
            elif value and f.name == 'volumes':
                setattr(
                    self,
                    f.name,
                    self._parse_volumes(value),
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
        valid = set(f.name for f in fields(cls))
        return cls(
            **{k: v for k, v in vars(args).items() if k in valid}
        )

    @functools.cached_property
    def vm_storage(self) -> pathlib.Path:
        return pathlib.Path(self.storage).joinpath(self.vm)

    @functools.cached_property
    def vm_disk(self) -> pathlib.Path:
        return pathlib.Path(
            self.storage
        ).joinpath(
            self.vm, self.vm
        ).with_suffix('.raw').resolve()

    @functools.cached_property
    def sock(self) -> pathlib.Path:
        return self.vm_disk.with_suffix('.sock')

    @functools.cached_property
    def pid(self) -> pathlib.Path:
        return self.vm_disk.with_suffix('.pid')

    @functools.cached_property
    def state_file(self) -> pathlib.Path:
        return self.storage / self.vm / 'run_config'

    def write(self) -> None:
        run_config = asdict(self)
        run_config.pop('force')
        run_config.pop('attach')
        state_file = self.state_file
        state_file.parent.mkdir(exist_ok=True)
        with (state_file).open('w') as f:
            json.dump(run_config, f, indent=4, cls=_JSONEncoder)

    @classmethod
    def read(cls, state_file: pathlib.Path) -> t.Self:
        if not state_file.is_file():
            raise ValueError(f'No such VM: {state_file.parent.name}')
        with state_file.open('rb') as f:
            if f.read(1) == b'\x80':
                f.seek(0)
                return cls(**pickle.load(f))
            f.seek(0)
            return cls(**json.load(f))


def parse_args(
        argv: list[str] | None = None
) -> tuple[t.Callable[[RunConfig], None], RunConfig]:
    vm_parser = argparse.ArgumentParser(add_help=False)
    vm_parser.add_argument(  # type: ignore[attr-defined]
        'vm',
        help='Name of VM'
    ).completer = storage_completer

    storage_parser = argparse.ArgumentParser(add_help=False)
    storage_parser.add_argument(
        '--storage',
        default=pathlib.Path.home() / 'vms' / 'storage',
        type=pathlib.Path,
        help='Path to vmstorage dir. Default: %(default)s',
    )

    run_common_parser = argparse.ArgumentParser(add_help=False)
    run_common_parser.add_argument(
        '--attach', '-a',
        action='store_true',
        default=False,
        help='Attach and to VM consolel and run in foreground',
    )
    run_common_parser.add_argument(
        '--gui',
        nargs='?',
        const=_DEFAULT_GUI_RESOLUTION,
        help=(
            'Enable GUI. Automatically enabled if VM is macOS. '
            'Defaults: %(const)s'
        ),
        metavar='WxH',
    )

    parents = [vm_parser, storage_parser]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    subparsers = parser.add_subparsers(dest='action', required=True)
    run_parser = subparsers.add_parser(
        'run',
        help=run.__doc__,
        parents=parents + [run_common_parser],
    )
    run_parser.set_defaults(
        action=run,
    )

    run_parser.add_argument(
        'src_image',
        nargs='?',
        help='Path to raw cloud image',
        type=pathlib.Path,
    )
    run_parser.add_argument(
        '--resize',
        default=Resize('+10'),
        help=(
            'Resize the disk in GB. Can be exact, or start with + '
            'to indicate a relative size change. Default: %(default)s'
        ),
        type=Resize,
    )
    run_parser.add_argument(
        '--cpus',
        default=2,
        help='Number of CPUs. Default %(default)s',
        type=int,
    )
    run_parser.add_argument(
        '--memory',
        default=2048,
        help='Amount of memory in MB. Default %(default)s',
        type=int,
    )
    run_parser.add_argument(
        '--user-data',
        default=pathlib.Path.home() / 'vms' / 'user-data',
        help='Path to cloud-init user_data file. Default %(default)s',
        type=pathlib.Path,
    )
    run_parser.add_argument(
        '--volume', '-v',
        dest='volumes',
        action='append',
        default=[],
        help=(
            'Volumes to mount into the VM. May be specified multiple '
            'times'
        ),
        metavar='src:dst',
    )
    run_parser.add_argument(
        '--mac',
        default=generate_laa_mac(),
        help='MAC address. Default: %(default)s',
    )
    run_parser.add_argument(
        '--force', '-f',
        action='store_true',
        default=False,
        help='Force recreation of VM',
    )
    run_parser.add_argument(
        '--state-file',
        type=pathlib.Path,
        help=argparse.SUPPRESS
    )

    subparsers.add_parser(
        'start',
        help=start.__doc__,
        parents=parents + [run_common_parser],
    ).set_defaults(
        action=start,
    )

    for func in (stop, rm, status):
        subparsers.add_parser(
            func.__name__,
            help=func.__doc__,
            parents=parents,
        ).set_defaults(
            action=func
        )

    subparsers.add_parser(  # type: ignore[attr-defined]
        'ls',
        help=ls.__doc__,
        parents=[storage_parser],
    ).add_argument(
        '--json',
        action='store_true'
    ).container.set_defaults(
        action=ls
    )

    if HAS_ARGCOMPLETE:
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)

    action = args.action
    if action is run:
        state_file = args.state_file
        if state_file and state_file.exists():
            run_config = RunConfig.read(args.state_file)
            run_config.force = args.force
            run_config.attach = args.attach
        else:
            run_config = RunConfig.from_argparse(args)
    elif action is start:
        run_config = RunConfig.read(
            RunConfig.from_argparse(args).state_file
        )
    elif action is ls:
        args.vm = ''
        run_config = RunConfig.from_argparse(args)

        action = functools.partial(
            ls,
            use_json=args.json
        )

    else:
        run_config = RunConfig.from_argparse(args)

    return action, run_config


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


def rm(run_config: RunConfig) -> None:
    """Remove a VM"""
    if run_config.pid.is_file():
        raise RuntimeError(
            f'{run_config.vm} is running'
        )
    shutil.rmtree(run_config.vm_disk.parent)


def ls(run_config: RunConfig, use_json=False) -> None:
    """List the status of VMs"""
    out = {}
    for vm_disk in run_config.storage.glob('*/*.raw'):
        vm = vm_disk.parent.name
        ip = True if use_json else False
        state = status(
            RunConfig.read(
                RunConfig(vm=vm, storage=run_config.storage).state_file
            ),
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
    """Get the status of a VM"""
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
    """Stop a VM"""
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


def start(run_config: RunConfig) -> None:
    """Start an existing VM"""
    pid = run_config.pid
    if pid.is_file():
        status(run_config, ip=False, ret=True)
        if pid.is_file():
            raise RuntimeError(
                f'{run_config.vm}[{int(pid.read_text())}] is already running'
            )

    is_mac = run_config.vm_storage.joinpath('AuxiliaryStorage').exists()

    if not run_config.vm_disk.is_file():
        raise ValueError(
            f'Could not locate disk image for {run_config.vm}'
        )

    efi = run_config.vm_disk.with_suffix('.efi')
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

    for src, dst in run_config.volumes:
        if not src.exists():
            src.mkdir()
        tag = dst.replace(os.sep, '-')
        cmd.extend([
            '--device',
            f'virtio-fs,sharedDir={src},mountTag={tag}',
        ])

    user_data = run_config.vm_storage / 'user-data'
    if user_data.is_file():
        cmd.extend(['--cloud-init', f'{user_data}'])

    if is_mac or run_config.gui:
        resolution = run_config.gui or _DEFAULT_GUI_RESOLUTION
        width, height = resolution.split('x')
        cmd.extend([
            '--device', 'virtio-input,keyboard',
            '--device', 'virtio-input,pointing',
            '--device', f'virtio-gpu,width={width},height={height}',
            '--gui',
        ])

    if run_config.attach:
        cmd.extend(['--device', 'virtio-serial,stdio'])

    popen_kwargs: dict[str, t.Any] = {'text': True}
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

    rc = p.poll()
    if rc is not None:
        stdout, _ = p.communicate()
        rest_sock.unlink(missing_ok=True)
        pid.unlink(missing_ok=True)
        raise RuntimeError(f'{run_config.vm} failed to start: {stdout}')

    print(json.dumps(
        status(run_config, ret=True),
        indent=4,
    ))

    if not run_config.attach:
        return

    try:
        p.wait()
    finally:
        rest_sock.unlink(missing_ok=True)
        pid.unlink(missing_ok=True)


def run(run_config: RunConfig) -> None:
    """Create and start a new VM"""
    is_new = not run_config.vm_disk.is_file()

    run_config.storage.mkdir(exist_ok=True)
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

            src_image = src_image.joinpath(
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

    if run_config.resize and (is_new or run_config.force):
        resize(run_config.vm_disk, run_config.resize)

    if run_config.force:
        efi = run_config.vm_disk.with_suffix('.efi')
        efi.unlink(missing_ok=True)

    mounts = []
    for src, dst in run_config.volumes:
        tag = dst.replace(os.sep, '-')
        mounts.append([
            tag, dst, 'virtiofs'
        ])

    user_data = run_config.user_data
    persistent_user_data = run_config.vm_storage / 'user-data'
    if mounts or (user_data and user_data.is_file()):
        with persistent_user_data.open('w') as f:
            if user_data and user_data.is_file():
                f.write(user_data.read_text())
            else:
                f.write('#cloud-config\n')
            if mounts:
                f.write(f'\nmounts: {json.dumps(mounts)}\n')
            f.flush()

    run_config.write()
    start(run_config)


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

    try:
        action, run_config = parse_args(argv)
        with _repair_stdin():
            action(run_config)
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
