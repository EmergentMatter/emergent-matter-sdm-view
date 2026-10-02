<a id="readme-top"></a>

# emergent-matter-sdm-view

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-033388.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-0055FF.svg)](https://www.python.org/downloads/)
[![Built with Blender](https://img.shields.io/badge/built%20with-Blender-E87D0D.svg)](https://www.blender.org/)
[![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)

Shared **Blender authoring + visualization layer** for Software Defined Matter CEMs.

A **bidirectional bridge** between JAX SDF cores and Blender. The goal is
SolidWorks-class in-context design authoring: any CEM that plugs in gets
parameter sliders, animation, cutaway views, parametric spreads, real-time
VDB iso-tuning, a GLSL ray-march viewer for `.sdm` parts, and an
annotation-dot backchannel for free.

## Install

This repo is **Blender-bound**: the `blender/` subpackage (the viewer,
cutaway, annotations, VDB import, and the rest of the catalog) only loads
inside Blender's own bundled Python, since Blender ships its own interpreter
and `bpy` is not a pip package. There is no addon-install step for
development: [`scripts/open_viewer.py`](scripts/open_viewer.py) registers
the viewer module straight from `src/` (see Quick start below).

Prerequisites:

- **Blender 4.2+**, for the GLSL ray-march viewer.
- A sibling **[`emergent-matter-sdm-core`](https://github.com/EmergentMatter/emergent-matter-sdm-core)**
  checkout, synced with `uv sync` there, for the SDF core the viewer imports
  and shells out to. Point `SDM_PYTHON=` at that checkout's
  `.venv/bin/python` if it is not a sibling directory of this repo.

`sdm-view` is published to the Software Defined Matter package index, not PyPI:

```bash
uv pip install sdm-view --index https://get.softwaredefinedmatter.com/simple
```

In a uv project, declare the index once and pin the package to it, so its name never resolves from public PyPI:

```toml
[[tool.uv.index]]
name = "em"
url = "https://get.softwaredefinedmatter.com/simple"
explicit = true

[tool.uv.sources]
sdm-view = { index = "em" }
```

For the pure-Python `io/` layer, and to run this repo's own tests and
linters, use [`uv`](https://docs.astral.sh/uv/) against a regular Python
3.13 environment:

```bash
git clone https://github.com/EmergentMatter/emergent-matter-sdm-view.git
cd emergent-matter-sdm-view
uv sync
```

That needs no sdm-core: the `io/` layer stays importable from a Python with
no solver installed. The tests that check this repo against sdm-core's
emitter skip unless it is importable, so add it to the same environment to
run them:

```bash
uv pip install -e ../emergent-matter-sdm-core
```

## Quick start: see a `.sdm` part

```bash
blender --factory-startup -P scripts/open_viewer.py -- /path/to/part.sdm
```

Ray-marches the part's SDF live in the viewport (no meshing), with every free
`Param` as an N-panel slider, generator-backed rebuilds, and authored
animations. Needs a sibling `emergent-matter-sdm-core` checkout (or
`SDM_PYTHON=` pointing at a venv that has it). `SDM_AUTOPLAY=1` plays the
first authored animation on open. Details:
[`src/sdm_view/blender/viewer/README.md`](src/sdm_view/blender/viewer/README.md).

## Layered API

```
src/sdm_view/
├── io/
│   ├── vdb_export.py
│   ├── annotation_state.py
│   ├── sdm_loader.py
│   └── sketch_state.py
└── blender/
    ├── cutaway.py
    ├── annotations.py
    ├── sketches.py
    ├── parts.py
    ├── project.py
    ├── vdb_import.py
    └── viewer/
```

Each module opens with a docstring stating what it provides. For the
`io`/`blender` split itself and how the pieces fit together, see
[`docs/architecture.md`](docs/architecture.md).

For `.sdm` parts there is a mesh-free path: the `viewer/` module ray-marches
the SDF directly in GLSL, so parameter scrubs are resolution-free and instant,
with nothing to re-voxelize at all.

Some catalog features (parameter auto-binding, dirty-flag auto-regen, pose
animation, a voxel-size knob, multi-worktree addon reload, viewport HUD/
graph/video overlays, parametric-spread showcase renders) have a design note
but no module yet. See the repo's issue tracker rather than a stub file for
what's planned.

## Why VDB-first

See [ADR 0001](docs/adr/0001-vdb-first-geometry-pipeline.md).

## Consumers

Any CEM that wants the authoring stack. A consumer defines its own frozen
`Parameters` dataclass and a `make_components(params)` callable, then composes
the `sdm_view.blender` feature modules into its addon's `register()`. The
load-bearing surface is `blender/{project,annotations,sketches,parts}` and
`io/{annotation_state,sketch_state,sdm_loader}`, so those signatures are the
ones to treat as a contract.

## Documentation

| Page | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | The `io`/`blender` package split and where the GLSL ray-march viewer fits |
| [`docs/adr/`](docs/adr/) | Architecture decision records, including [ADR 0001](docs/adr/0001-vdb-first-geometry-pipeline.md) on the VDB-first geometry pipeline |
| [`src/sdm_view/blender/viewer/README.md`](src/sdm_view/blender/viewer/README.md) | The GLSL ray-march viewer: how it works, prerequisites, known limitations |
| [STYLE.md](STYLE.md) / [CONTRIBUTING.md](CONTRIBUTING.md) | House style and how a change ships |

Those pages document this library. For the wider SDM ecosystem and the
other repositories in it, see
[`emergent-matter-sdm`](https://github.com/EmergentMatter/emergent-matter-sdm).

## Built with

| | |
|---|---|
| [Blender](https://www.blender.org/) | The host application. The ray-march viewer, VDB volume import, cutaway, and annotation tooling all run inside its bundled Python |
| [NumPy](https://numpy.org/) | Grid construction for the VDB export path |
| [OpenVDB](https://www.openvdb.org/) | Blender's bundled module; `io/vdb_export.py` writes the SDF as a `.vdb` grid ([ADR 0001](docs/adr/0001-vdb-first-geometry-pipeline.md)) |
| [uv](https://docs.astral.sh/uv/) | Packaging and the locked dev environment |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how a change ships: the
changeset a pull request needs, what counts as major, minor, or patch,
and how a release is cut. [STYLE.md](STYLE.md) is the house style for
code, tests, and docs, and it wins over habit.

By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Support

Questions and usage help go to
[Discussions](https://github.com/EmergentMatter/emergent-matter-sdm/discussions);
bugs and feature requests go to
[Issues](https://github.com/EmergentMatter/emergent-matter-sdm-view/issues).
See [SUPPORT.md](SUPPORT.md) for what is and is not supported.

For security reports, do not open a public issue. Follow
[SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Acknowledgments

- The VDB-first geometry pipeline ([ADR 0001](docs/adr/0001-vdb-first-geometry-pipeline.md))
  and the GLSL ray-march viewer both descend from earlier Emergent Matter
  internal Blender addons. The viewer was merged into this repo on
  2026-07-07 with its full history.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
