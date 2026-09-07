-- Parameterized nonlinear Landau-damping pilot for multi-case FNO training.
-- Defaults retain the strict single-case numerical setup. Environment values
-- select k/A without mechanically rewriting this input file.

local Vlasov = G0.Vlasov

local pi = math.pi
local epsilon0, mu0 = 1.0, 1.0
local mass_elc, charge_elc = 1.0, -1.0
local vt = 1.0

local alpha = tonumber(os.getenv("GKYL_ALPHA")) or 0.10
local k0 = tonumber(os.getenv("GKYL_K0")) or 0.35
local Nx, Nvx = 64, 64
local Lx = 2.0 * pi / k0
local vx_max = 6.0 * vt

local t_end = tonumber(os.getenv("GKYL_T_END")) or 40.0
local num_frames = tonumber(os.getenv("GKYL_NUM_FRAMES")) or 8000
local distribution_stride = tonumber(os.getenv("GKYL_DISTRIBUTION_FRAME_STRIDE")) or 2000
-- dx scales as 1/k.  Scale cflFrac with k so all parameter cases retain an
-- approximately common internal dt~0.0047, safely below dt_out=0.005.
local cfl_fraction = tonumber(os.getenv("GKYL_CFL_FRAC")) or (0.525 * k0 / 0.35)

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
