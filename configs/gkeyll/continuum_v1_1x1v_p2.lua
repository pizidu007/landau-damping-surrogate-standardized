-- GPU-only normalized 1x1v Vlasov--Maxwell input for continuum_v1.
--
-- Normalization: omega_pe = v_th = lambda_D = n0 = 1, so GKYL_K0 is
-- K = k_phys*lambda_D and time is tau = omega_pe*t.  Physical (n0,T0)
-- values are scale metadata and do not change this normalized trajectory.

local Vlasov = G0.Vlasov

local pi = math.pi
local epsilon0, mu0 = 1.0, 1.0
local mass_elc, charge_elc = 1.0, -1.0
local vt = 1.0

local alpha = tonumber(os.getenv("GKYL_ALPHA")) or 0.10
local k0 = tonumber(os.getenv("GKYL_K0")) or 0.35
local Nx = tonumber(os.getenv("GKYL_NX")) or 96
local Nvx = tonumber(os.getenv("GKYL_NV")) or 192
local vx_max = tonumber(os.getenv("GKYL_VMAX")) or 8.0
local t_end = tonumber(os.getenv("GKYL_T_END")) or 60.0
local num_frames = tonumber(os.getenv("GKYL_NUM_FRAMES")) or 3000
local distribution_stride =
  tonumber(os.getenv("GKYL_DISTRIBUTION_FRAME_STRIDE")) or 5
local cfl_fraction = tonumber(os.getenv("GKYL_CFL_FRAC")) or 0.50
local Lx = 2.0 * pi / k0

assert(k0 > 0.0, "GKYL_K0 must be positive")
assert(alpha >= 0.0 and alpha < 1.0, "GKYL_ALPHA must be in [0,1)")
assert(Nx >= 4 and Nvx >= 8, "GKYL_NX/GKYL_NV are too small")
assert(vx_max > 0.0, "GKYL_VMAX must be positive")
assert(t_end > 0.0 and num_frames >= 1, "invalid output schedule")
assert(distribution_stride >= 1, "distribution stride must be positive")
assert(cfl_fraction > 0.0 and cfl_fraction <= 1.0, "invalid CFL fraction")

vlasovApp = Vlasov.App.new {
  tEnd = t_end,
  nFrame = num_frames,
  distributionFrameStride = distribution_stride,
  fieldEnergyCalcs = num_frames,
  integratedMomentCalcs = num_frames,
  integratedL2fCalcs = num_frames,
  dtFailureTol = 1.0e-4,
  numFailuresMax = 20,

  lower = { 0.0 },
  upper = { Lx },
  cells = { Nx },
  cflFrac = cfl_fraction,
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
