-- Strict single-case reproduction of Huang et al. (PNAS, 2025).
--
-- Physics values are fixed to the paper's nonlinear Landau-damping case:
--   k=0.35, A=0.10, x in [0, 2*pi/k], v in [-6, 6], Nx=Nv=64, t in [0, 40].
-- The official Gkeyll Landau regression uses the same Vlasov-Ampere setup,
-- RK3, and a p2 serendipity DG basis.  The paper does not state its DG order,
-- so p2 serendipity is the one explicitly recorded numerical assumption here.
-- cflFrac=0.53 keeps the measured p2 CUDA step below the paper's 0.005
-- sampling interval for the entire nonlinear run.  A full t=40 audit at
-- cflFrac=0.573 found that dt grew from 0.00500 to 0.00536 and therefore
-- skipped 407 output triggers; 0.53 includes margin against that growth.
--
-- The optional environment overrides are only for a short preflight run.  The
-- formal run must leave them unset, giving 8000 intervals (dt_out=0.005).

local Vlasov = G0.Vlasov

local pi = math.pi
local epsilon0, mu0 = 1.0, 1.0
local mass_elc, charge_elc = 1.0, -1.0
local vt = 1.0

local alpha = 0.10
local k0 = 0.35
local Nx, Nvx = 64, 64
local Lx = 2.0 * pi / k0
local vx_max = 6.0 * vt

local t_end = tonumber(os.getenv("GKYL_T_END")) or 40.0
local num_frames = tonumber(os.getenv("GKYL_NUM_FRAMES")) or 8000

vlasovApp = Vlasov.App.new {
  tEnd = t_end,
  nFrame = num_frames,
  fieldEnergyCalcs = num_frames,
  integratedMomentCalcs = num_frames,
  integratedL2fCalcs = num_frames,
  dtFailureTol = 1.0e-4,
  numFailuresMax = 20,

  lower = { 0.0 },
  upper = { Lx },
  cells = { Nx },
  cflFrac = 0.53,
  basis = "serendipity",
  polyOrder = 2,
  timeStepper = "rk3",
  decompCuts = { 1 },
  periodicDirs = { 1 },

  elc = Vlasov.Species.new {
    modelID = G0.Model.Default,
    charge = charge_elc,
    mass = mass_elc,
    lower = { -vx_max },
    upper = { vx_max },
    cells = { Nvx },
    numInit = 1,
    projections = {
      {
        projectionID = G0.Projection.Func,
        init = function (t, xn)
          local x, vx = xn[1], xn[2]
          local maxwellian = math.exp(-vx * vx / (2.0 * vt * vt)) /
            math.sqrt(2.0 * pi * vt * vt)
          return (1.0 + alpha * math.cos(k0 * x)) * maxwellian
        end
      }
    },
    evolve = true,
    diagnostics = {
      G0.Moment.M0,
      G0.Moment.M1,
      G0.Moment.M2,
      G0.Moment.M3
    }
  },

  field = Vlasov.Field.new {
    epsilon0 = epsilon0,
    mu0 = mu0,
    init = function (t, xn)
      local x = xn[1]
      local Ex = -alpha * math.sin(k0 * x) / k0
      return Ex, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    end,
    evolve = true,
    elcErrorSpeedFactor = 0.0,
    mgnErrorSpeedFactor = 0.0
  }
}

vlasovApp:run()
