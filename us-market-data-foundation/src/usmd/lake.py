"""Guarded external lake writes with atomic files and no internal-disk fallback."""
from __future__ import annotations
import contextlib
import fcntl
import importlib.util
import io
import json
import math
import os
import stat
from pathlib import Path
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('storage_preflight', PROJECT_ROOT / 'scripts/check_storage.py')
_preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preflight)
check_storage = _preflight.check


class Lake:
    def __init__(self, config=None):
        if config is None:
            config = json.loads((PROJECT_ROOT / 'config/storage.json').read_text())
        elif isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        self.config = config
        self.root = Path(config['external_data_root'])
        self.preflight()

    def preflight(self):
        return check_storage(self.config)

    def usage_bytes(self, allocated=True):
        total = 0
        for base, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
            for filename in files:
                p = Path(base) / filename
                if p.is_symlink():
                    continue
                stat = p.stat()
                total += stat.st_blocks * 512 if allocated else stat.st_size
        return total

    def _parts(self, relative_path):
        p = Path(relative_path)
        if p.is_absolute() or not p.parts or any(x in ('..', '.') for x in p.parts):
            raise ValueError('Only safe relative lake paths are allowed')
        return p.parts

    def _check_budget(self, size):
        cluster = self.config.get('external_allocation_bytes_observed', 1048576)
        # Include temporary new file alongside old published file plus directory slack.
        reserve = math.ceil(size / cluster) * cluster + 4 * cluster
        if self.usage_bytes() + reserve > self.config['external_phase1_budget_gib'] * 1024**3:
            raise ValueError('Lake allocated-size budget exceeded; old files preserved')

    @contextlib.contextmanager
    def _parent_fd(self, parts, create=False):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(self.root, flags)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    def write_bytes(self, relative_path, content):
        if not isinstance(content, bytes):
            raise TypeError('content must be bytes')
        parts = self._parts(relative_path)
        self.preflight()
        self._check_budget(len(content))
        with self._parent_fd(parts, create=True) as parent:
            name = parts[-1]
            try:
                if os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode & 0o170000 == 0o120000:
                    raise ValueError('Refusing a symlink target')
            except FileNotFoundError:
                pass
            tmp = '.' + name + '.' + uuid.uuid4().hex + '.tmp'
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                self.preflight()
                os.replace(tmp, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(tmp, dir_fd=parent)
                except FileNotFoundError:
                    pass
        return str(self.root.joinpath(*parts))

    def write_json(self, relative_path, obj):
        return self.write_bytes(relative_path, (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n').encode())

    def safe_work_dir(self, relative_path):
        """Create an external-only scratch directory; caller holds writer_lock.

        Check the budget between bounded operations using this directory. Never
        use this for a local fallback if the configured volume is unavailable.
        """
        parts = self._parts(relative_path)
        self.preflight()
        self._check_budget(0)
        with self._parent_fd((*parts, '_guard'), create=True) as parent:
            path = self.root.joinpath(*parts)
            if not os.path.samestat(os.fstat(parent), path.stat(follow_symlinks=False)):
                raise ValueError('Scratch directory changed while opening')
        return path

    def write_generated(self, relative_path, producer, estimated_size=0):
        """Atomically publish a file produced by a streaming writer on the lake.

        The producer receives a reserved temporary Path on the same filesystem.
        Failure leaves the previous publication intact. The caller holds the
        writer lock; producer must not create unbounded intermediate files.
        """
        parts = self._parts(relative_path)
        self.preflight()
        self._check_budget(estimated_size)
        with self._parent_fd(parts, create=True) as parent:
            name = parts[-1]
            try:
                if stat.S_ISLNK(os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode):
                    raise ValueError('Refusing a symlink target')
            except FileNotFoundError:
                pass
            tmp = '.' + name + '.' + uuid.uuid4().hex + '.tmp'
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            os.close(fd)
            temp_path = self.root.joinpath(*parts[:-1], tmp)
            try:
                producer(temp_path)
                self.preflight()
                if not os.path.samestat(os.fstat(parent), temp_path.parent.stat(follow_symlinks=False)):
                    raise ValueError('Publication directory changed while writing')
                fd = os.open(tmp, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise ValueError('Producer must create a regular file')
                    os.fsync(fd)
                finally:
                    os.close(fd)
                self._check_budget(0)  # Temporary output is already counted.
                os.replace(tmp, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                for leftover in (tmp, '._' + tmp):
                    try:
                        os.unlink(leftover, dir_fd=parent)
                    except FileNotFoundError:
                        pass
        return str(self.root.joinpath(*parts))

    def write_parquet(self, relative_path, frame):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False, compression='zstd')
        return self.write_bytes(relative_path, buffer.getvalue())

    @contextlib.contextmanager
    def writer_lock(self):
        self.preflight()
        with self._parent_fd(('_writer.lock',)) as parent:
            fd = os.open('_writer.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
            finally:
                os.close(fd)

    @contextlib.contextmanager
    def reader_lock(self):
        self.preflight()
        with self._parent_fd(('_writer.lock',)) as parent:
            fd = os.open('_writer.lock', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                yield
            finally:
                os.close(fd)
