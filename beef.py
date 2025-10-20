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

import abc
import argparse
import collections.abc as c
import contextlib
import ctypes
import fcntl
import functools
import json
import os
import pathlib
import platform
import random
import re
import shutil
import socket
import struct
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

_MBR_BOOTABLE_FLAG = 0x80

clonefile: t.Callable[[bytes, bytes, int], int]
if sys.platform == 'darwin':
    _LIBC = ctypes.CDLL(None)
    clonefile = _LIBC.clonefile
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, pathlib.Path):
            return str(o)
        if isinstance(o, Resolution):
            return str(o)

        return super().default(o)


def storage_completer(
        prefix: str,
        parsed_args: argparse.Namespace,
        **kwargs
) -> c.Iterator:
    storage = pathlib.Path(parsed_args.storage)
    return (p.name for p in storage.glob(f'{prefix}*') if p.is_dir())


def generate_laa_mac() -> str:
    first_byte = random.randint(0x00, 0xFF) & 0b11111100 | 0b00000010
    mac = [first_byte] + [random.randint(0x00, 0xFF) for _ in range(5)]
    return ':'.join(f'{b:02x}' for b in mac)


class Resize(str):
    def __new__(cls, value):
        if int(value) < 0:
            raise ValueError('negative values not supported')
        return str(value)


@dataclass(frozen=True, slots=True)
class Resolution:
    width: int
    height: int

    def __init__(self, value: str | t.Self | None = None):
        match value:
            case None:
                width, height = 1024, 800
            case Resolution():
                width, height = value.width, value.height
            case str():
                w, h = value.split('x')
                width, height = int(w), int(h)
            case _:
                raise TypeError(
                    f'Invalid resolution type: {value.__class__.__name__}'
                )

        object.__setattr__(self, 'width', width)
        object.__setattr__(self, 'height', height)

    def __str__(self) -> str:
        return f'{self.width}x{self.height}'


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
    ) -> list[tuple[pathlib.Path, str]]: ...

    @t.overload
    def _parse_volumes(
            self,
            value: list[tuple[pathlib.Path, str]]
    ) -> list[tuple[pathlib.Path, str]]: ...

    def _parse_volumes(self, value):
        volumes = []
        if value == [None]:
            return volumes

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
    def control_sock(self) -> pathlib.Path:
        return self.vm_storage / 'control.sock'

    @functools.cached_property
    def pid(self) -> pathlib.Path:
        return self.vm_disk.with_suffix('.pid')

    @functools.cached_property
    def state_file(self) -> pathlib.Path:
        return self.storage / self.vm / 'config.json'

    def asdict(self) -> dict[str, object]:
        run_config = asdict(self)
        run_config.pop('force')
        run_config.pop('attach')
        return run_config

    def write(self) -> None:
        run_config = self.asdict()
        state_file = self.state_file
        state_file.parent.mkdir(exist_ok=True)
        with (state_file).open('w') as f:
            json.dump(run_config, f, indent=4, cls=_JSONEncoder)

    @classmethod
    def read(cls, state_file: pathlib.Path) -> t.Self:
        if not state_file.is_file():
            raise ValueError(f'No such VM: {state_file.parent.name}')
        with state_file.open('rb') as f:
            return cls(**json.load(f))


@contextlib.contextmanager
def _make_control_sock(run_config: RunConfig):
    """Create a Unix socket connection to the VM's control socket"""
    pid = run_config.pid
    if not pid.parent.is_dir():
        raise ValueError(f'No such VM: {run_config.vm}')

    if not pid.is_file():
        raise RuntimeError(f'{run_config.vm} is not running')

    control_sock = run_config.control_sock
    if not control_sock.is_socket():
        raise RuntimeError('control sock is missing')

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(control_sock))
        yield sock


class HypervisorBackend(abc.ABC):
    _bridge_interface: str

    @abc.abstractmethod
    def verify(self) -> None:
        """Verify hypervisor is available"""

    def get_vm_ip(self, mac: str) -> str | None:
        """Get VM IP address from MAC address"""
        arp_re = re.compile(rf'^\S+ \(([^)]+)\) at {mac}', flags=re.M)

        cmd = ['arp', '-an']
        if bridge_interface := getattr(self, '_bridge_interface', None):
            cmd.extend(['-i', bridge_interface])

        ip_match = None
        for _ in range(30):
            try:
                p = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if ip_match := arp_re.search(p.stdout):
                    break
                time.sleep(1)
                continue
            except subprocess.SubprocessError:
                time.sleep(1)
                continue

        if ip_match:
            return ip_match.group(1)
        return None

    @abc.abstractmethod
    def get_status(self, run_config: RunConfig) -> dict[str, object]:
        """Get VM status"""

    @abc.abstractmethod
    def stop_vm(self, run_config: RunConfig) -> None:
        """Stop a running VM"""

    @abc.abstractmethod
    def start_vm(self, run_config: RunConfig) -> subprocess.Popen:
        """Start a VM and return Popen object"""
        if not run_config.vm_disk.is_file():
            raise ValueError(
                f'Could not locate disk image for {run_config.vm}'
            )

    @abc.abstractmethod
    def _get_mount_entry(self, tag: str, dst: str) -> list[str]:
        """Get mount entry format for this hypervisor"""

    def prepare_vm_storage(self, run_config: RunConfig) -> None:
        """Prepare VM storage (cloud-init, etc.)"""
        mounts = []
        for src, dst in run_config.volumes:
            tag = dst.replace(os.sep, '-')
            mounts.append(self._get_mount_entry(tag, dst))

        user_data = run_config.user_data
        user_data_file = run_config.vm_storage / 'user-data'

        if mounts or (user_data and user_data.is_file()):
            with user_data_file.open('w') as f:
                if user_data and user_data.is_file():
                    f.write(user_data.read_text())
                else:
                    f.write('#cloud-config\n')
                if mounts:
                    f.write(f'\nmounts: {json.dumps(mounts)}\n')
                f.flush()

    @abc.abstractmethod
    def prepare_vm_metadata(self, run_config: RunConfig) -> pathlib.Path:
        """Prepare VM metadata (bundles, boot artifacts, etc.) and return
        actual disk path to clone
        """
        src_image = run_config.src_image
        if not src_image:
            raise ValueError('src_image is required')
        return src_image

    @abc.abstractmethod
    def clone_disk(
            self,
            src: pathlib.Path,
            dst: pathlib.Path
    ) -> None:
        """Clone disk image from src to dst"""


class VfkitBackend(HypervisorBackend):
    _bridge_interface = 'bridge100'

    def verify(self) -> None:
        if not shutil.which('vfkit'):
            raise RuntimeError(
                'vfkit not found, please install vfkit '
                '(https://github.com/crc-org/vfkit)'
            )

    def get_vm_ip(self, mac: str) -> str | None:
        mac = re.sub(r'(^|:)0', r'\1', mac)
        return super().get_vm_ip(mac)

    def get_status(self, run_config: RunConfig) -> dict[str, object]:
        data: dict[str, object]
        try:
            with _make_control_sock(run_config) as sock:
                sock.sendall(textwrap.dedent('''
                    GET /vm/state HTTP/1.1
                    Host: localhost

                ''').lstrip().encode('iso8859-1'))
                resp = sock.recv(1024).decode()
        except (RuntimeError, socket.error):
            data = {"state": "Stopped"}
            if run_config.pid.is_file():
                run_config.pid.unlink(missing_ok=True)
                run_config.control_sock.unlink(missing_ok=True)
        else:
            raw_data = json.loads(resp.partition('\r\n\r\n')[2])
            state = raw_data.get('state') or 'Unknown'
            if state.startswith('VirtualMachineState'):
                state = state.replace('VirtualMachineState', '')
            data = {
                'state': state,
                'pid': int(run_config.pid.read_text()),
            }

        return data

    def stop_vm(self, run_config: RunConfig) -> None:
        with _make_control_sock(run_config) as sock:
            sock.sendall(textwrap.dedent('''
                POST /vm/state HTTP/1.1
                Host: localhost
                Content-Type: application/json
                Content-Length: 17

                {"state": "Stop"}
            ''').lstrip().encode('iso8859-1'))
            if sock.recv(21) != b'HTTP/1.1 202 Accepted':
                raise RuntimeError(
                    f'Could not issue stop to {run_config.vm}'
                )
        run_config.pid.unlink()
        run_config.control_sock.unlink()

    def _get_mount_entry(self, tag: str, dst: str) -> list[str]:
        return [tag, dst, 'virtiofs']

    def prepare_vm_metadata(self, run_config: RunConfig) -> pathlib.Path:
        src_image = super().prepare_vm_metadata(run_config)

        if run_config.force:
            efi = run_config.vm_disk.with_suffix('.efi')
            efi.unlink(missing_ok=True)

        is_bundle = (
            src_image.is_dir() and src_image.suffix == '.bundle'
        )

        if not is_bundle:
            return src_image

        for file in ('AuxiliaryStorage', 'HardwareModel',
                     'MachineIdentifier'):
            shutil.copy2(
                src_image / file,
                run_config.vm_storage
            )
        return src_image / 'Disk.img'

    def clone_disk(
            self,
            src: pathlib.Path,
            dst: pathlib.Path
    ) -> None:
        rc = clonefile(bytes(src), bytes(dst), 0)
        if rc == -1:
            dst.unlink(missing_ok=True)
            raise OSError(f'Could not clone {src} to {dst}')

    def start_vm(self, run_config: RunConfig) -> subprocess.Popen:
        super().start_vm(run_config)

        is_mac = run_config.vm_storage.joinpath('AuxiliaryStorage').exists()

        efi = run_config.vm_disk.with_suffix('.efi')
        control_sock = run_config.control_sock
        if control_sock.is_file():
            control_sock.unlink()

        cmd = [
            'vfkit',
            '--cpus', f'{run_config.cpus}',
            '--memory', f'{run_config.memory}',
            '--device', f'virtio-blk,path={run_config.vm_disk}',
            '--device', f'virtio-net,nat,mac={run_config.mac}',
            '--device', 'virtio-rng',
            '--restful-uri', f'unix:{control_sock}',
        ]

        if is_mac:
            vm_storage = run_config.vm_storage
            machine_identifier_path = vm_storage / 'MachineIdentifier'
            cmd.extend([
                '--bootloader',
                (
                    'macos,'
                    f'machineIdentifierPath={machine_identifier_path},'
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
                src.mkdir(parents=True, exist_ok=True)
            tag = dst.replace(os.sep, '-')
            cmd.extend([
                '--device',
                f'virtio-fs,sharedDir={src},mountTag={tag}',
            ])

        user_data = run_config.vm_storage / 'user-data'
        if user_data.is_file():
            cmd.extend(['--cloud-init', f'{user_data}'])

        if is_mac or run_config.gui:
            resolution = run_config.gui or Resolution()
            cmd.extend([
                '--device', 'virtio-input,keyboard',
                '--device', 'virtio-input,pointing',
                '--device', (
                    f'virtio-gpu,width={resolution.width},'
                    f'height={resolution.height}'
                ),
                '--gui',
            ])

        if run_config.attach:
            cmd.extend(['--device', 'virtio-serial,stdio'])

        popen_kwargs: dict[str, t.Any] = {
            'text': True,
            'env': os.environ | {'TMPDIR': str(run_config.vm_storage)},
        }
        if not run_config.attach:
            popen_kwargs.update({
                'start_new_session': True,
                'stdin': subprocess.PIPE,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
            })

        p = subprocess.Popen(cmd, **popen_kwargs)
        p.poll()

        return p


class QemuBackend(HypervisorBackend):
    _bridge_interface = 'virbr0'

    def __init__(self):
        self._machine_type = 'q35'
        self._qemu_binary = 'qemu'
        self._bridge_helper = 'qemu-bridge-helper'
        self._ovmf_code = pathlib.Path('OVMF_CODE.fd')
        self._ovmf_vars_template = pathlib.Path('OVMF_VARS.fd')

    def _detect_qemu_binary(self) -> str:
        """Detect appropriate QEMU binary for the system architecture"""
        arch = platform.machine().lower()

        if arch in {'x86_64', 'amd64'}:
            binary = 'qemu-system-x86_64'
            self._machine_type = 'q35'
        elif arch in {'aarch64', 'arm64'}:
            binary = 'qemu-system-aarch64'
            self._machine_type = 'virt'
        else:
            raise RuntimeError(f'Unsupported architecture: {arch}')

        if not shutil.which(binary):
            raise RuntimeError(f'{binary} not found')

        return binary

    def _detect_bridge_helper(self) -> str:
        """Detect qemu-bridge-helper location"""
        common_paths = (
            '/usr/lib/qemu/qemu-bridge-helper',
            '/usr/libexec/qemu-bridge-helper',
        )

        for path in common_paths:
            if pathlib.Path(path).exists():
                return path

        helper = shutil.which('qemu-bridge-helper')
        if helper:
            return helper

        raise RuntimeError(
            'qemu-bridge-helper not found. Please install qemu-bridge-helper '
            'and ensure /etc/qemu/bridge.conf contains "allow virbr0"'
        )

    def _detect_ovmf(self) -> None:
        """Detect OVMF firmware files"""
        ovmf_variants = (
            ('OVMF_CODE_4M.fd', 'OVMF_VARS_4M.fd'),
            ('OVMF_CODE.fd', 'OVMF_VARS.fd'),
        )

        for code_file, vars_file in ovmf_variants:
            code_path = pathlib.Path('/usr/share/OVMF') / code_file
            vars_path = pathlib.Path('/usr/share/OVMF') / vars_file
            if code_path.exists() and vars_path.exists():
                self._ovmf_code = code_path
                self._ovmf_vars_template = vars_path
                return

        raise RuntimeError('OVMF not found, please install ovmf')

    def verify(self) -> None:
        self._qemu_binary = self._detect_qemu_binary()
        self._bridge_helper = self._detect_bridge_helper()
        self._detect_ovmf()

        if not shutil.which('genisoimage'):
            raise RuntimeError(
                'genisoimage not found, please install genisoimage'
            )

        helper_path = pathlib.Path(self._bridge_helper)
        if not (helper_path.stat().st_mode & 0o4000):
            raise RuntimeError(
                f'qemu-bridge-helper at {self._bridge_helper} does not have '
                f'setuid bit set. Run: sudo chmod u+s {self._bridge_helper}'
            )

        bridge_conf = pathlib.Path('/etc/qemu/bridge.conf')
        if not bridge_conf.exists():
            raise RuntimeError(
                '/etc/qemu/bridge.conf not found. '
                'Run: echo "allow virbr0" | sudo tee /etc/qemu/bridge.conf'
            )

        if 'allow virbr0' not in bridge_conf.read_text():
            raise RuntimeError(
                '/etc/qemu/bridge.conf does not contain "allow virbr0". '
                'Run: echo "allow virbr0" | sudo tee -a /etc/qemu/bridge.conf'
            )

    def _qmp_command(
            self,
            sock: socket.socket,
            command: str,
            arguments: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Send QMP command and return response"""
        cmd: dict[str, object] = {'execute': command}
        if arguments:
            cmd['arguments'] = arguments

        sock.sendall((json.dumps(cmd) + '\n').encode())

        response = b''
        for chunk in iter(functools.partial(sock.recv, 4096), b''):
            response += chunk
            if b'\n' in chunk:
                break

        lines = response.decode().strip().split('\n')
        for line in lines:
            if stripped_line := line.strip():
                data = json.loads(stripped_line)
                if 'return' in data or 'error' in data:
                    return data

        return {}

    def get_status(self, run_config: RunConfig) -> dict[str, object]:
        data: dict[str, object]
        try:
            with _make_control_sock(run_config) as sock:
                sock.recv(4096)

                self._qmp_command(sock, 'qmp_capabilities')
                result = self._qmp_command(sock, 'query-status')

                if 'return' in result:
                    return_data = t.cast(dict[str, str], result['return'])
                    qemu_status: str = return_data.get('status') or 'Unknown'
                    state_map = {
                        'running': 'Running',
                        'paused': 'Paused',
                        'shutdown': 'Stopped',
                        'inmigrate': 'Running',
                    }
                    data = {
                        'state': state_map.get(
                            qemu_status,
                            qemu_status.title()
                        )
                    }
                else:
                    data = {'state': 'Unknown'}
        except (RuntimeError, socket.error, json.JSONDecodeError):
            data = {"state": "Stopped"}
            if run_config.pid.is_file():
                run_config.pid.unlink(missing_ok=True)
                run_config.control_sock.unlink(missing_ok=True)
        else:
            data['pid'] = int(run_config.pid.read_text())

        return data

    def stop_vm(self, run_config: RunConfig) -> None:
        with _make_control_sock(run_config) as sock:
            sock.recv(4096)

            self._qmp_command(sock, 'qmp_capabilities')
            result = self._qmp_command(sock, 'system_powerdown')

            if 'error' in result:
                raise RuntimeError(
                    f'Could not issue stop to {run_config.vm}: '
                    f'{result["error"]}'
                )

        run_config.pid.unlink()
        run_config.control_sock.unlink()

    def _get_mount_entry(self, tag: str, dst: str) -> list[str]:
        return [tag, dst, '9p', 'trans=virtio', '0', '0']

    def _generate_cloud_init_iso(self, run_config: RunConfig) -> None:
        """Generate cloud-init ISO from user-data file"""
        user_data_file = run_config.vm_storage / 'user-data'
        if not user_data_file.is_file():
            return

        seed_iso = run_config.vm_storage / 'cloud-init.iso'
        meta_data = run_config.vm_storage / 'meta-data'
        meta_data.write_text('instance-id: iid-local01\n')

        subprocess.run(
            [
                'genisoimage',
                '-output', str(seed_iso),
                '-volid', 'cidata',
                '-joliet', '-rock',
                str(user_data_file),
                str(meta_data),
            ],
            check=True,
            capture_output=True,
        )

    def prepare_vm_metadata(self, run_config: RunConfig) -> pathlib.Path:
        src_image = super().prepare_vm_metadata(run_config)

        if run_config.force:
            ovmf_vars = run_config.vm_storage / self._ovmf_vars_template.name
            ovmf_vars.unlink(missing_ok=True)

        return src_image

    def clone_disk(
            self,
            src: pathlib.Path,
            dst: pathlib.Path
    ) -> None:
        with src.open('rb') as src_fd:
            with dst.open('wb') as dst_fd:
                try:
                    FICLONE = fcntl.FICLONE  # type: ignore[missing-attribute]
                    fcntl.ioctl(dst_fd.fileno(), FICLONE, src_fd.fileno())
                except OSError as e:
                    dst.unlink(missing_ok=True)
                    raise OSError(
                        f'Could not clone {src} to {dst}: {e}. '
                        'Reflink cloning is required. Ensure your filesystem '
                        'supports reflinks (btrfs, xfs with reflink enabled)'
                    )

    def start_vm(self, run_config: RunConfig) -> subprocess.Popen:
        super().start_vm(run_config)

        self._generate_cloud_init_iso(run_config)

        control_sock = run_config.control_sock
        if control_sock.is_file():
            control_sock.unlink()

        ovmf_code = self._ovmf_code
        ovmf_vars = run_config.vm_storage / self._ovmf_vars_template.name

        if not ovmf_vars.exists():
            shutil.copy2(self._ovmf_vars_template, ovmf_vars)

        bridge_helper = self._bridge_helper
        cmd = [
            self._qemu_binary,
            '-name', run_config.vm,
            '-machine', f'{self._machine_type},accel=kvm',
            '-cpu', 'host',
            '-smp', str(run_config.cpus),
            '-m', str(run_config.memory),
            '-drive', f'file={run_config.vm_disk},format=raw,if=virtio',
            '-netdev', f'bridge,id=net0,br=virbr0,helper={bridge_helper}',
            '-device', f'virtio-net-pci,netdev=net0,mac={run_config.mac}',
            '-device', 'virtio-rng-pci',
            '-qmp', f'unix:{control_sock},server,nowait',
        ]

        if ovmf_code.exists():
            cmd.extend([
                '-drive', f'if=pflash,format=raw,readonly=on,file={ovmf_code}',
                '-drive', f'if=pflash,format=raw,file={ovmf_vars}',
            ])

        seed_iso = run_config.vm_storage / 'cloud-init.iso'
        if seed_iso.is_file():
            cmd.extend([
                '-drive', f'file={seed_iso},format=raw,if=virtio,readonly=on',
            ])

        for src, dst in run_config.volumes:
            if not src.exists():
                src.mkdir(parents=True, exist_ok=True)
            tag = dst.replace(os.sep, '-')
            cmd.extend([
                '-fsdev',
                f'local,id=fsdev{tag},path={src},security_model=passthrough',
                '-device',
                f'virtio-9p-pci,fsdev=fsdev{tag},mount_tag={tag}',
            ])

        if run_config.gui:
            resolution = run_config.gui or Resolution()
            cmd.extend([
                '-device', (
                    f'virtio-vga,xres={resolution.width},'
                    f'yres={resolution.height}'
                ),
                '-display', 'sdl',
            ])
        else:
            cmd.extend(['-display', 'none'])

        if run_config.attach:
            cmd.extend(['-serial', 'stdio'])

        popen_kwargs: dict[str, t.Any] = {
            'text': True,
        }

        if not run_config.attach:
            popen_kwargs.update({
                'start_new_session': True,
                'stdin': subprocess.PIPE,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
            })

        p = subprocess.Popen(cmd, **popen_kwargs)
        p.poll()

        return p


@functools.cache
def _get_backend() -> HypervisorBackend:
    """Get the appropriate hypervisor backend for the current platform"""
    if sys.platform == 'darwin':
        return VfkitBackend()
    elif sys.platform == 'linux':
        return QemuBackend()
    else:
        raise RuntimeError(f'Unsupported platform: {sys.platform}')


def _settable_parser(defaults: bool = True) -> argparse.ArgumentParser:
    arguments: list[tuple[tuple[str, ...], dict[str, t.Any]]] = [
        (
            ('--resize',),
            {
                'default': Resize('+10'),
                'help': (
                    'Resize the disk in GB. Can be exact, or start with + '
                    'to indicate a relative size change'
                ),
                'type': Resize,
            },
        ),
        (
            ('--cpus',),
            {
                'default': 2,
                'help': 'Number of CPUs',
                'type': int,
            },
        ),
        (
            ('--memory',),
            {
                'default': 2048,
                'help': 'Amount of memory in MB',
                'type': int,
            },
        ),
        (
            ('--user-data',),
            {
                'default': os.getenv(
                    'BEEF_USER_DATA',
                    pathlib.Path.home() / 'vms' / 'user-data',
                ),
                'help': 'Path to cloud-init user_data file',
                'type': pathlib.Path,
            },
        ),
        (
            ('--volume', '-v'),
            {
                'dest': 'volumes',
                'action': 'append',
                'default': [],
                'help': (
                    'Volumes to mount into the VM. May be specified multiple '
                    'times'
                ),
                'metavar': 'src:dst',
            },
        ),
        (
            ('--mac',),
            {
                'default': generate_laa_mac(),
                'help': 'MAC address',
            },
        ),
    ]

    parser = argparse.ArgumentParser(add_help=False)
    for args, kwargs in arguments:
        if not defaults:
            if isinstance(kwargs['default'], list):
                kwargs['nargs'] = '?'
                kwargs['const'] = None
            kwargs['default'] = None
        elif kwargs.get('default'):
            kwargs['help'] += '. Default: %(default)s'
        parser.add_argument(*args, **kwargs)
    return parser


def parse_args(
        argv: list[str] | None = None
) -> tuple[t.Callable[[RunConfig], None], RunConfig]:
    vm_parser = argparse.ArgumentParser(add_help=False)
    vm_parser.add_argument(  # type: ignore[attr-defined]
        'vm',
        help='Name of VM',
    ).completer = storage_completer

    storage_parser = argparse.ArgumentParser(add_help=False)
    storage_parser.add_argument(
        '--storage',
        default=os.getenv(
            'BEEF_STORAGE',
            pathlib.Path.home() / 'vms' / 'storage',
        ),
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
        type=Resolution,
        const=Resolution(),
        help=(
            'Enable GUI. Automatically enabled if VM is macOS. '
            'Defaults: %(const)s'
        ),
        metavar='WxH',
    )

    parents = [vm_parser, storage_parser]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f'%(prog)s {__version__}',
    )

    subparsers = parser.add_subparsers(dest='action', required=True)
    run_parser = subparsers.add_parser(
        'run',
        help=run.__doc__,
        parents=parents + [_settable_parser(), run_common_parser],
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
        '--force', '-f',
        action='store_true',
        default=False,
        help='Force recreation of VM',
    )
    run_parser.add_argument(
        '--state-file',
        type=pathlib.Path,
        help=argparse.SUPPRESS,
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
            action=func,
        )

    subparsers.add_parser(  # type: ignore[attr-defined]
        'ls',
        help=ls.__doc__,
        parents=[storage_parser],
    ).add_argument(
        '--json',
        action='store_true',
    ).container.set_defaults(
        action=ls,
    )

    subparsers.add_parser(
        'set',
        help=reconfigure.__doc__,
        parents=parents + [_settable_parser(defaults=False)],
    ).set_defaults(
        action=reconfigure,
    )

    subparsers.add_parser(
        'get',
        help=get.__doc__,
        parents=parents,
    ).set_defaults(
        action=get,
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
    elif getattr(action, '_requires_state', False):
        run_config = RunConfig.read(
            RunConfig.from_argparse(args).state_file
        )
        with contextlib.suppress(AttributeError):
            run_config.attach = args.attach
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


def get_vm_ip(mac: str) -> str | None:
    backend = _get_backend()
    return backend.get_vm_ip(mac)


@dataclass(kw_only=True, slots=True)
class _Partition:
    bootable: bool
    type: int
    start_lba: int
    sectors: int


def _parse_partition_entry(entry: bytes) -> _Partition:
    return _Partition(
        bootable=entry[0] == _MBR_BOOTABLE_FLAG,
        type=entry[4],
        start_lba=struct.unpack_from("<I", entry, 8)[0],
        sectors=struct.unpack_from("<I", entry, 12)[0],
    )


def _has_valid_mbr(disk: pathlib.Path) -> bool:
    with disk.open('rb') as f:
        data = f.read(512)

    if data[510:512] != b'\x55\xAA':
        return False

    for i in range(4):
        offset = i * 16
        entry = data[446 + offset:446 + offset + 16]
        partition = _parse_partition_entry(entry)
        if partition.type != 0 and partition.sectors != 0:
            return True

    return False


def resize(vm_disk: pathlib.Path, size: Resize) -> None:
    current = vm_disk.stat().st_size
    if size.isdigit():
        new = int(size) * 1024**3
    else:
        new = current + int(size) * 1024**3
    if new < current:
        raise ValueError('resize cannot be smaller than disk image')
    os.truncate(vm_disk, new)


def _check_running(run_config: RunConfig) -> None:
    pid = run_config.pid
    if pid.is_file():
        status(run_config, ip=False, ret=True)
        if pid.is_file():
            raise RuntimeError(
                f'{run_config.vm}[{int(pid.read_text())}] is running'
            )


def _requires_state(
        func: t.Callable[..., t.Any]
) -> t.Callable[..., t.Any]:
    func._requires_state = True  # type: ignore[attr-defined]
    return func


def rm(run_config: RunConfig) -> None:
    """Remove a VM"""
    _check_running(run_config)
    shutil.rmtree(run_config.vm_storage)


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


@_requires_state
def status(
        run_config: RunConfig,
        ip: bool = True,
        ret: bool = False
) -> dict[str, object] | t.NoReturn:
    """Get the status of a VM"""
    backend = _get_backend()
    data = backend.get_status(run_config)

    if ip and data.get('state') != 'Stopped':
        if run_config.mac is None:
            raise ValueError('MAC address not set')
        data['ipAddress'] = get_vm_ip(run_config.mac)

    rc = 0 if data.get('state') != 'Stopped' else 2

    if ret:
        return data

    print(json.dumps(data, indent=4))
    sys.exit(rc)


def stop(run_config: RunConfig) -> None:
    """Stop a VM"""
    backend = _get_backend()
    backend.stop_vm(run_config)


def _write_user_data(run_config: RunConfig) -> pathlib.Path | None:
    backend = _get_backend()
    backend.prepare_vm_storage(run_config)
    persistent_user_data = run_config.vm_storage / 'user-data'
    if persistent_user_data.is_file():
        return persistent_user_data
    return None


@_requires_state
def get(run_config: RunConfig) -> None:
    """Get VM configuration"""
    print(
        json.dumps(
            run_config.asdict(),
            indent=4,
            cls=_JSONEncoder,
        )
    )


def reconfigure(run_config: RunConfig) -> None:
    """Reconfigure VM"""
    _check_running(run_config)

    current = RunConfig.read(run_config.state_file)

    for f in fields(run_config):
        value = getattr(run_config, f.name)
        if value is None:
            continue
        if f.name == 'resize' and value and current.resize:
            if value[0] == '+':
                value = Resize(int(current.resize) + int(value))
        setattr(current, f.name, value)

    if current.resize:
        resize(current.vm_disk, current.resize)

    if run_config.user_data or run_config.volumes is not None:
        _write_user_data(current)

    current.write()


@_requires_state
def start(run_config: RunConfig) -> None:
    """Start an existing VM"""
    _check_running(run_config)

    backend = _get_backend()
    p = backend.start_vm(run_config)

    run_config.pid.write_text(f'{p.pid}\n')

    control_sock = run_config.control_sock
    for _ in range(3):
        if control_sock.is_socket():
            break
        time.sleep(1)

    if p.poll() is not None:
        stdout, _ = p.communicate()
        control_sock.unlink(missing_ok=True)
        run_config.pid.unlink(missing_ok=True)
        raise RuntimeError(f'{run_config.vm} failed to start: {stdout}')

    print(
        json.dumps(
            status(run_config, ret=True),
            indent=4,
        )
    )

    if run_config.attach:
        try:
            p.wait()
        finally:
            run_config.control_sock.unlink(missing_ok=True)
            run_config.pid.unlink(missing_ok=True)


def run(run_config: RunConfig) -> None:
    """Create and start a new VM"""
    run_config.storage.mkdir(exist_ok=True)
    run_config.vm_storage.mkdir(exist_ok=True)
    if run_config.force and run_config.src_image:
        run_config.vm_disk.unlink(missing_ok=True)

    if run_config.vm_disk.is_file():
        raise ValueError(f'{run_config.vm_disk} already exists')

    backend = _get_backend()
    src_image = backend.prepare_vm_metadata(run_config)

    if not _has_valid_mbr(src_image):
        raise ValueError(
            f'{src_image} does not appear to raw disk image'
        )

    backend.clone_disk(src_image, run_config.vm_disk)

    if run_config.resize:
        resize(run_config.vm_disk, run_config.resize)

    _write_user_data(run_config)

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


def _verify_hypervisor() -> None:
    try:
        backend = _get_backend()
        backend.verify()
    except RuntimeError as e:
        print(f'{e}', file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    _verify_hypervisor()

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
