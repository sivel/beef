# beef
The Beefcake macOS Virtualization lifecycle manager

This project uses [vfkit](https://github.com/crc-org/vfkit).

## About

`beef` adds some syntactic sugar on top of managing the lifecycle of VMs running through `vfkit` such as:

1. `run`, `stop`, `ls`, `status`, and `rm` commands
1. cloning and extending source raw image files
1. state saving for re-launching an existing VM by name
1. abstraction of complex arguments required for configuring VMs
1. abstraction of running Linux and macOS VMs

## Example use

### Linux

```
$ curl -LO https://download.fedoraproject.org/pub/fedora/linux/releases/42/Cloud/aarch64/images/Fedora-Cloud-Base-AmazonEC2-42-1.1.aarch64.raw.xz
$ unxz -T0 Fedora-Cloud-Base-AmazonEC2-42-1.1.aarch64.raw.xz
$ beef run -d fedora42 Fedora-Cloud-Base-AmazonEC2-42-1.1.aarch64.raw
{
    "canHardStop": true,
    "canPause": true,
    "canResume": false,
    "canStart": false,
    "canStop": true,
    "state": "VirtualMachineStateRunning",
    "ipAddress": "192.168.64.56",
    "pid": 96456
}
```

### macOS

This example assumes that you have a `VM.bundle` created by the `InstallationTool` provided in the [sample project](https://developer.apple.com/documentation/virtualization/running-macos-in-a-virtual-machine-on-apple-silicon).

Note: macOS requires running with `--gui`

```
$ beef run --gui macos VM.Bundle
```
