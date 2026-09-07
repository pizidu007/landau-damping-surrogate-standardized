
# Third-party notices

## Immediate upstream project

This project is a modified and substantially extended derivative of:

- **Project:** `kaneman777/landau-damping-surrogate`
- **Repository:** https://github.com/kaneman777/landau-damping-surrogate
- **Pinned reference commit:** `9490dbff6b322bf7f43bbcff49af48bb9ff65fd7`
- **Local reference:** `reference/upstream/`

The upstream tracked `LICENSE.md` contains GNU GPL v3. Its README mentions MIT,
which conflicts with that license file; this derivative follows the actual GPL
v3 license file and retains the full attribution chain.

## Original 1D electrostatic PIC code

The immediate upstream project and the `landau_surrogate.pic` subpackage derive
from:

- **Project:** 1D-PIC (electrostatic)
- **Original author:** Antoine Tavant
- **Upstream repository:** https://github.com/antoinelpp/1d-pic-electrostatic

The supplied source repository contains the GNU General Public License version
3. This release therefore retains GPL-3.0 for the combined distributed work.
The previous work-in-progress README mentioned MIT, but that statement
conflicted with the actual license file and has not been carried forward.

Relative to the immediate upstream, this downstream project adds the packaged
x-v HDF5 workflow, conditional 2D FNO snapshot and autoregressive models,
multi-step training, positivity objectives, conservative cache processing,
Poisson/energy closure diagnostics, reproducible artifact manifests, and a
standardized source/data/model/result layout. The PIC code was adapted to use
package-qualified imports. See `reference/DIFFERENCES.md`.
