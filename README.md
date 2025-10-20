# beef
The Beefcake Virtualization lifecycle manager

A simple VM lifecycle manager for macOS and Linux.

- **macOS**: Uses [vfkit](https://github.com/crc-org/vfkit) with the Virtualization.framework
- **Linux**: Uses QEMU with KVM

## About

`beef` adds some syntactic sugar on top of managing VM lifecycles:

1. `run`, `start`, `stop`, `rm`, `status`, `ls`, `get` and `set` commands
1. cloning and extending source raw image files
1. state saving for re-launching an existing VM by name
1. abstraction of complex arguments required for configuring VMs
1. abstraction of running Linux and macOS VMs

## Setup

### macOS

Install vfkit from https://github.com/crc-org/vfkit or using whatever method you prefer

### Linux

**Filesystem Requirements**: Linux requires a filesystem that supports reflink cloning (copy-on-write). Supported filesystems:
- **btrfs** (native support)
- **xfs** (requires `mkfs.xfs -m reflink=1` when formatting)

#### Ubuntu/Debian

```bash
# Install packages
sudo apt install qemu-system-x86 libvirt-daemon-system ovmf genisoimage

# For ARM systems, use qemu-system-arm instead:
# sudo apt install qemu-system-arm libvirt-daemon-system ovmf genisoimage

# Configure qemu-bridge-helper
echo "allow virbr0" | sudo tee /etc/qemu/bridge.conf
sudo chmod u+s /usr/lib/qemu/qemu-bridge-helper
```

#### Fedora/RHEL

```bash
# Install packages
sudo dnf install qemu-system-x86 libvirt-daemon edk2-ovmf genisoimage

# For ARM systems, use qemu-system-aarch64 instead:
# sudo dnf install qemu-system-aarch64 libvirt-daemon edk2-ovmf genisoimage

# Enable and start libvirt networking
sudo systemctl enable --now libvirtd
sudo virsh net-start default
sudo virsh net-autostart default

# Configure qemu-bridge-helper
echo "allow virbr0" | sudo tee /etc/qemu/bridge.conf
sudo chmod u+s /usr/libexec/qemu-bridge-helper
```

**Note**: The `chmod u+s` command sets the setuid bit on qemu-bridge-helper, which is required for unprivileged bridge networking. This is a standard configuration for QEMU bridge networking.

## Example use

### Running a Linux VM

This example works on both macOS and Linux hosts.

```
$ curl -LO https://download.fedoraproject.org/pub/fedora/linux/releases/42/Cloud/aarch64/images/Fedora-Cloud-Base-AmazonEC2-42-1.1.aarch64.raw.xz
$ unxz -T0 Fedora-Cloud-Base-AmazonEC2-42-1.1.aarch64.raw.xz
$ beef run fedora42 Fedora-Cloud-Base-AmazonEC2-42-1.1.aarch64.raw
{
    "state": "Running",
    "ipAddress": "192.168.64.56",
    "pid": 96456
}
```

### Running a macOS VM

This example is macOS host only.

This example assumes that you have a `VM.bundle`. This can be created using [https://github.com/Code-Hex/vz/tree/main/example/macOS](https://github.com/Code-Hex/vz/tree/main/example/macOS).

Note: macOS VMs require running with `--gui`, if omitted, it is implicitly assumed.

```
$ beef run --gui -- macos VM.bundle
```
