"""Opt-in recovery for unbounded MIPS task probes authored by OpenCode.

The ``make-mips-interpreter`` task legitimately runs persistent programs, but
an exploratory shell probe must still return to the agent. This guard handles
only observed Node probes whose shell requested a short SIGINT shutdown, piped
an endless VM into ``tail``, kept polling for a frame after a bounded VM
exited, or directly launched a native Doom image while expecting it to crash.
It revalidates exact process ancestry and command evidence inside the exact E2B
sandbox and signals only the stuck leaf. It never signals OpenCode, the
sandbox, the trainer, the rollout, or a process group.
"""
import inspect
import json
import os
from pathlib import Path
import re
import signal


TASK_PREFIX = 'harbor-opencode-make-mips-interpreter-'


def _chain(expected, proc_root):
    records = []
    for index, item in enumerate(expected['chain']):
        process = proc_root / str(item['pid'])
        stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        name = (process / 'comm').read_text().strip()
        if stat[19] != item['start_ticks'] or stat[0] == 'Z' or name != item['name']:
            return []
        if index + 1 < len(expected['chain']) and int(stat[1]) != expected['chain'][index + 1]['pid']:
            return []
        records.append((process, stat))
    return records


def overdue_mips_node(expected, proc_root=Path('/proc')):
    """Confirm a known unbounded Node probe after its intended short window."""
    records = _chain(expected, proc_root)
    names = [item['name'] for item in expected['chain']]
    if not records or names[0] != 'node' or names[-1] != 'opencode' or any(
            name != 'bash' for name in names[1:-1]):
        return False
    child_args = records[0][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    parent_args = records[1][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if not child_args or Path(os.fsdecode(child_args[0])).name != 'node':
        return False
    child = b' '.join(child_args)
    quadratic_probe = (
        b'node -e' in child
        and b'for (let i = 0; i < 200000; i++)' in child
        and b'for (const a of vals) for (const b of vals)' in child
    )
    if len(parent_args) < 3 and not quadratic_probe:
        return False
    command = parent_args[-1] if parent_args else b''
    known_vm = (b'vm.js' in child
                or (b'node -e' in child and b'while(true)' in child)
                or quadratic_probe)
    if not known_vm:
        return False
    elapsed = (float((proc_root / 'uptime').read_text().split()[0])
               - int(records[0][1][19]) / os.sysconf('SC_CLK_TCK'))
    # The observed arithmetic self-test appends 200k values and then executes
    # their Cartesian product (~40B iterations). It is an exploratory probe,
    # not the task workload, and has no practical completion time.
    if quadratic_probe:
        return elapsed > 60
    # The agent explicitly requested a short run and SIGINT, but the synchronous
    # VM prevented Node from servicing the signal.  Preserve 30 seconds of grace.
    sleep = re.search(rb'(?:^|[;&|] *)sleep +([0-9]+(?:[.][0-9]+)?)', command)
    if b'kill -INT' in command and b'wait' in command and sleep:
        return elapsed > float(sleep[1]) + 30
    # The task's VM executes a persistent Doom/MIPS program.  A direct foreground
    # invocation (including output pipelines) has no natural EOF; explicit GNU
    # timeout wrappers are handled by node_timeout_guard instead.
    if b'vm.js' in command and b'timeout ' not in command:
        return elapsed > 60
    return False


def overdue_mips_frame_wait(expected, proc_root=Path('/proc')):
    """Confirm an unbounded frame wait left after a bounded VM run.

    The observed shell starts ``timeout N node vm.js`` in the background, then
    waits for ``/tmp/frame.bmp`` without bounding that second loop. If the VM
    exits without producing a frame, the shell otherwise waits forever. Match
    that concrete command and preserve the declared timeout plus grace;
    generic shell loops remain outside this recovery policy.
    """
    records = _chain(expected, proc_root)
    names = [item['name'] for item in expected['chain']]
    if not records or names != ['bash', 'opencode']:
        return False
    args = records[0][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if len(args) < 3 or Path(os.fsdecode(args[0])).name != 'bash' or args[1] != b'-c':
        return False
    command = args[-1]
    timeout = re.search(
        rb'(?:^|[;&|] *)timeout +([0-9]+(?:[.][0-9]+)?)([smhd]?) +node +[^;&|]*vm[.]js',
        command,
    )
    missing_frame_wait = b'while [ ! -f /tmp/frame.bmp ]' in command
    repeated_frame_poll = (
        b'while true; do if [ -f /tmp/frame.bmp ]' in command
        and b'sleep ' in command
    )
    if not timeout or not (missing_frame_wait or repeated_frame_poll):
        return False
    duration = float(timeout[1]) * {
        b'': 1, b's': 1, b'm': 60, b'h': 3600, b'd': 86400,
    }[timeout[2]]
    elapsed = (float((proc_root / 'uptime').read_text().split()[0])
               - int(records[0][1][19]) / os.sysconf('SC_CLK_TCK'))
    return elapsed > duration + 30


def overdue_mips_native_crash_probe(expected, proc_root=Path('/proc')):
    """Confirm a persistent native Doom run launched only to obtain a core."""
    records = _chain(expected, proc_root)
    names = [item['name'] for item in expected['chain']]
    if not records or names != ['doom_x86', 'bash', 'bash', 'opencode']:
        return False
    child_args = records[0][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    parent_args = records[1][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    tool_args = records[2][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if child_args != [b'./doom_x86'] or len(parent_args) < 3 or len(tool_args) < 3:
        return False
    if os.readlink(records[0][0] / 'cwd') != '/tmp/opencode/native':
        return False
    parent_command, tool_command = parent_args[-1], tool_args[-1]
    if (parent_command != b'ulimit -c unlimited; ./doom_x86 >/dev/null 2>&1'
            or b'rm -f core' not in tool_command
            or b'ld -static' not in tool_command
            or b'./doom_x86 >/dev/null 2>&1' not in tool_command):
        return False
    elapsed = (float((proc_root / 'uptime').read_text().split()[0])
               - int(records[0][1][19]) / os.sysconf('SC_CLK_TCK'))
    return elapsed > 60


def overdue_mips_qemu_head_probe(expected, proc_root=Path('/proc')):
    """Confirm a persistent QEMU Doom probe piped to a finite ``head``."""
    records = _chain(expected, proc_root)
    names = [item['name'] for item in expected['chain']]
    if not records or names != ['qemu-mipsel', 'bash', 'opencode']:
        return False
    child_args = records[0][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    tool_args = records[1][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    known_children = (
        [b'qemu-mipsel', b'-strace', b'/app/doomgeneric_mips'],
        [b'qemu-mipsel', b'-D', b'd.log', b'-d', b'page,ftrace', b'doom_qemu'],
    )
    if child_args not in known_children or len(tool_args) < 3:
        return False
    if os.readlink(records[0][0] / 'cwd') != '/tmp/opencode/qpatch':
        return False
    command = tool_args[-1]
    required = (
        b'qemu-mipsel -strace /app/doomgeneric_mips 2>&1 | head -6',
        b'qemu-mipsel -D d.log -d page,ftrace doom_qemu 2>&1 | head',
    )
    if not all(marker in command for marker in required):
        return False
    elapsed = (float((proc_root / 'uptime').read_text().split()[0])
               - int(records[0][1][19]) / os.sysconf('SC_CLK_TCK'))
    return elapsed > 60


def signal_overdue_mips_node(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not overdue_mips_node(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        return {'action': 'SIGKILL', 'pid': expected['chain'][0]['pid'],
                'start_ticks': expected['chain'][0]['start_ticks'],
                'reason': 'unbounded_make_mips_node_probe'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def signal_overdue_mips_frame_wait(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not overdue_mips_frame_wait(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return {'action': 'SIGTERM', 'pid': expected['chain'][0]['pid'],
                'start_ticks': expected['chain'][0]['start_ticks'],
                'reason': 'unbounded_frame_wait_after_bounded_node_exit'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def signal_overdue_mips_native_crash_probe(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not overdue_mips_native_crash_probe(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return {'action': 'SIGTERM', 'pid': expected['chain'][0]['pid'],
                'start_ticks': expected['chain'][0]['start_ticks'],
                'reason': 'unbounded_native_doom_crash_probe'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def signal_overdue_mips_qemu_head_probe(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not overdue_mips_qemu_head_probe(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return {'action': 'SIGTERM', 'pid': expected['chain'][0]['pid'],
                'start_ticks': expected['chain'][0]['start_ticks'],
                'reason': 'unbounded_qemu_doom_head_probe'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_candidates(previous, current):
    """Require the same task, tool and process ancestry on two observations."""
    if (not previous or not current.get('trial', '').startswith(TASK_PREFIX)
            or previous.get('sandbox_id') != current.get('sandbox_id')
            or previous.get('phase') != current.get('phase')
            or current.get('phase') != 'agent_or_setup'):
        return []
    before, after = previous.get('remote', {}), current.get('remote', {})
    starts = {tool.get('start_ms') for tool in before.get('running_tools', [])
              if tool.get('tool') == 'bash' and tool.get('start_ms') is not None}
    if not any(tool.get('tool') == 'bash' and tool.get('start_ms') in starts
               for tool in after.get('running_tools', [])):
        return []
    old = {process['Pid']: process for process in before.get('processes', [])}
    new = {process['Pid']: process for process in after.get('processes', [])}
    node_present = any(process.get('Name') == 'node' for process in new.values())
    candidates = []
    for process in new.values():
        leaf_name = process.get('Name')
        if leaf_name not in ('node', 'bash', 'doom_x86', 'qemu-mipsel') or (process.get('elapsed_s') or 0) <= 30:
            continue
        chain = [process]
        for _ in range(3):
            parent = new.get(chain[-1].get('PPid'))
            if not parent:
                break
            chain.append(parent)
            if parent.get('Name') == 'opencode':
                break
        names = [item.get('Name') for item in chain]
        if names[-1:] != ['opencode']:
            continue
        if leaf_name == 'node' and any(name != 'bash' for name in names[1:-1]):
            continue
        if leaf_name == 'doom_x86' and names != ['doom_x86', 'bash', 'bash', 'opencode']:
            continue
        if leaf_name == 'qemu-mipsel' and names != ['qemu-mipsel', 'bash', 'opencode']:
            continue
        # A frame-wait shell is recoverable only after the bounded VM process
        # has disappeared. While any Node process remains, leave the tool alone.
        if leaf_name == 'bash' and (node_present or names != ['bash', 'opencode']):
            continue
        if not all(item.get('start_ticks') is not None
                   and item['start_ticks'] == old.get(item['Pid'], {}).get('start_ticks')
                   and item.get('PPid') == old.get(item['Pid'], {}).get('PPid')
                   for item in chain):
            continue
        kind = ('node' if leaf_name == 'node' else
                'native_crash_probe' if leaf_name == 'doom_x86' else
                'qemu_head_probe' if leaf_name == 'qemu-mipsel' else
                'frame_wait_shell')
        candidates.append({'kind': kind,
                           'chain': [
            {'pid': int(item['Pid']), 'start_ticks': item['start_ticks'], 'name': item['Name']}
            for item in chain]})
    return candidates


def recovery_command(expected):
    """Send identity evidence only; command text remains inside the sandbox."""
    payload = {'kind': expected.get('kind', 'node'),
               'chain': [{'pid': int(item['pid']),
                          'start_ticks': str(int(item['start_ticks'])),
                          'name': item['name']} for item in expected['chain']]}
    functions = {
        'node': (overdue_mips_node, signal_overdue_mips_node),
        'frame_wait_shell': (
            overdue_mips_frame_wait, signal_overdue_mips_frame_wait,
        ),
        'native_crash_probe': (
            overdue_mips_native_crash_probe,
            signal_overdue_mips_native_crash_probe,
        ),
        'qemu_head_probe': (
            overdue_mips_qemu_head_probe,
            signal_overdue_mips_qemu_head_probe,
        ),
    }
    validate, recover = functions.get(payload['kind'], functions['node'])
    return ("python3 - <<'RECOVER'\nimport os, re, signal, json\nfrom pathlib import Path\n"
            + inspect.getsource(_chain) + '\n'
            + inspect.getsource(validate) + '\n'
            + inspect.getsource(recover) + '\n'
            + f"expected = json.loads({json.dumps(payload)!r})\n"
            + f"print(json.dumps({recover.__name__}(expected)))\nRECOVER")
