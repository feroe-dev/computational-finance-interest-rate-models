package net.finmath.lecture.computationalfinance.project;

import java.util.Locale;
import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.lecture.computationalfinance.project.measure.ForwardMeasure;
import net.finmath.lecture.computationalfinance.project.measure.Measure;
import net.finmath.lecture.computationalfinance.project.measure.RidingTheCurveMeasure;
import net.finmath.lecture.computationalfinance.project.measure.SpotMeasure;
import net.finmath.lecture.computationalfinance.project.measure.TerminalMeasure;
import net.finmath.lecture.computationalfinance.project.measure.ZeroDriftMeasure;
import net.finmath.lecture.computationalfinance.project.statespace.DisplacedLogNormalStateSpaceTransform;
import net.finmath.lecture.computationalfinance.project.statespace.LogNormalStateSpaceTransform;
import net.finmath.lecture.computationalfinance.project.statespace.NormalStateSpaceTransform;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterpolation;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.RandomVariableFromArrayFactory;
import net.finmath.montecarlo.interestrate.LIBORMonteCarloSimulationFromLIBORModel;
import net.finmath.montecarlo.interestrate.models.LIBORMarketModelFromCovarianceModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORVolatilityModelFourParameterExponentialForm;
import net.finmath.montecarlo.interestrate.products.Caplet;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretizationFromArray;

public class LIBORMarketModelPluginTest {
    
    // -------------------------------------------------------------------------
	// Shared model parameters
	// -------------------------------------------------------------------------

	/** Number of Monte Carlo paths. */
	private static final int NUMBER_OF_PATHS = 50000;

	/** Random seeds — one fixed seed per measure type for consistency within a measure. */
	private static final int SEED_SPOT     = 3141;
	private static final int SEED_TERMINAL = 3142;
    private static final int SEED_FORWARD10 = 3143;
    private static final int SEED_FORWARD14 = 3144;
    private static final int SEED_FORWARD18 = 3145;
    private static final int SEED_FORWARD6   = 3146;
	private static final int SEED_ZERO_DRIFT = 3147;
	private static final int SEED_RIDING     = 3148;



	/** Semi-annual tenor and simulation step (0 to 10 years, 20 periods). */
	private static final double PERIOD_LENGTH = 0.5;
	private static final double TIME_HORIZON  = 10.0;

	/** We work with an initial flat forward rate */
	private static final double FLAT_FORWARD_RATE = 0.03;

	/**
	 * Volatility parameters for the four-parameter exponential form
	 * \( \sigma(\tau) = (a + b\,\tau)\,e^{-c\,\tau} + d \).
	 * With b = 0 and d = 0 this simplifies to \( \sigma(\tau) = a\,e^{-c\,\tau} \).
	 */
	private static final double VOLATILITY_PARAMETER_A = 0.20;  // multiplied with initial forward rate for normal dynamics
	private static final double VOLATILITY_PARAMETER_B = 0.0;
	private static final double VOLATILITY_PARAMETER_C = 0.25;
	private static final double VOLATILITY_PARAMETER_D = 0.0;

	/** 
	 * Correlation matrix is the factor reduced matrix of the exponential decay
	 * correlation model \( \rho_{i,j} = \exp(-c\,|T_i - T_j|) \).
	*/
	/** Correlation decay parameter. */
	private static final double CORRELATION_DECAY = 0.10;

	/** Number of Brownian factors. */
	private static final int NUMBER_OF_FACTORS = 5;

	// =========================================================================
    // Ex1 — Measure plug-in: Forward Measure
    // =========================================================================
	@Test
	public void ex1_forwardMeasure() throws CalculationException {

        final double fixingTime  = 4.5;
		final double paymentTime = 5.0;
        final double notional = 10000.0;
        final double strikeITM = 0.01;
        final double strikeOTM = 0.05;

        final int fixingTimeIndex = (int) (fixingTime / PERIOD_LENGTH);

        final Caplet capletITM = new Caplet(fixingTime, paymentTime-fixingTime, strikeITM);
        final Caplet capletATM = new Caplet(fixingTime, paymentTime-fixingTime, FLAT_FORWARD_RATE);
        final Caplet capletOTM = new Caplet(fixingTime, paymentTime-fixingTime, strikeOTM);
        
        // Numerical valuation with normal state space transform
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward6     = buildNewSimulation(new ForwardMeasure(6), new NormalStateSpaceTransform(), SEED_FORWARD6);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward10    = buildNewSimulation(new ForwardMeasure(10), new NormalStateSpaceTransform(), SEED_FORWARD10);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward14    = buildNewSimulation(new ForwardMeasure(14), new NormalStateSpaceTransform(), SEED_FORWARD14);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward18    = buildNewSimulation(new ForwardMeasure(18), new NormalStateSpaceTransform(), SEED_FORWARD18);  

        final double ITMforward6 = capletITM.getValue(simulationPluginForward6) * notional;
        final double ATMforward6 = capletATM.getValue(simulationPluginForward6) * notional;
        final double OTMforward6 = capletOTM.getValue(simulationPluginForward6) * notional;


        final double ITMforward10 = capletITM.getValue(simulationPluginForward10) * notional;
        final double ATMforward10 = capletATM.getValue(simulationPluginForward10) * notional;
        final double OTMforward10 = capletOTM.getValue(simulationPluginForward10) * notional;

        final double ATMforward14 = capletATM.getValue(simulationPluginForward14) * notional;
        final double OTMforward14 = capletOTM.getValue(simulationPluginForward14) * notional;
        final double ITMforward14 = capletITM.getValue(simulationPluginForward14) * notional;

        final double ATMforward18 = capletATM.getValue(simulationPluginForward18) * notional;
        final double OTMforward18 = capletOTM.getValue(simulationPluginForward18) * notional;
        final double ITMforward18 = capletITM.getValue(simulationPluginForward18) * notional;

        // Numerical valuation with lognormal state space transform
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward6Lognormal     = buildNewSimulation(new ForwardMeasure(6), new LogNormalStateSpaceTransform(), SEED_FORWARD6);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward10Lognormal    = buildNewSimulation(new ForwardMeasure(10), new LogNormalStateSpaceTransform(), SEED_FORWARD10);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward14Lognormal    = buildNewSimulation(new ForwardMeasure(14), new LogNormalStateSpaceTransform(), SEED_FORWARD14);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginForward18Lognormal    = buildNewSimulation(new ForwardMeasure(18), new LogNormalStateSpaceTransform(), SEED_FORWARD18);

        final double ITMforward6Lognormal = capletITM.getValue(simulationPluginForward6Lognormal) * notional;
        final double ATMforward6Lognormal = capletATM.getValue(simulationPluginForward6Lognormal) * notional;
        final double OTMforward6Lognormal = capletOTM.getValue(simulationPluginForward6Lognormal) * notional;

        final double ITMforward10Lognormal = capletITM.getValue(simulationPluginForward10Lognormal) * notional;
        final double ATMforward10Lognormal = capletATM.getValue(simulationPluginForward10Lognormal) * notional;
        final double OTMforward10Lognormal = capletOTM.getValue(simulationPluginForward10Lognormal) * notional;

        final double ITMforward14Lognormal = capletITM.getValue(simulationPluginForward14Lognormal) * notional;
        final double ATMforward14Lognormal = capletATM.getValue(simulationPluginForward14Lognormal) * notional;
        final double OTMforward14Lognormal = capletOTM.getValue(simulationPluginForward14Lognormal) * notional;

        final double ITMforward18Lognormal = capletITM.getValue(simulationPluginForward18Lognormal) * notional;
        final double ATMforward18Lognormal = capletATM.getValue(simulationPluginForward18Lognormal) * notional;
        final double OTMforward18Lognormal = capletOTM.getValue(simulationPluginForward18Lognormal) * notional;

        // Analytic valuation for normal case
        final double normalizedVolatilityParameterA = FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A; // multiplied with initial forward rate for normal dynamics
        final double ITMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, strikeITM, fixingTimeIndex, normalizedVolatilityParameterA, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double ATMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingTimeIndex, normalizedVolatilityParameterA, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double OTMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, strikeOTM, fixingTimeIndex, normalizedVolatilityParameterA, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

        // Analytic valuation for lognormal case
        final double ITMtheoreticalBlack = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeITM, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double ATMtheoreticalBlack = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double OTMtheoreticalBlack = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeOTM, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

        System.out.println("=".repeat(80));
        System.out.println("  EXERCISE 1/2 — Measure Plug-In | Forward Measure");
        System.out.println("  Caplet: Fixing = 4.5 yr, Payment = 5.0 yr | Flat forward curve at 3%");
        System.out.println("=".repeat(80));
        System.out.println("");

        System.out.println("--- ITM Caplet (strike = 1 %) | Theoretical value: " + ITMtheoretical + " under Bachelier, " + ITMtheoreticalBlack + " under Black");
        System.out.println("-".repeat(70));
        System.out.printf(Locale.US, "%-18s  %12s  %12s  %12s  %12s%n", "State Space", "Forward 6", "Forward 10", "Forward 14", "Forward 18");
        System.out.println("-".repeat(70));
        System.out.printf(Locale.US, "%-18s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "Normal", ITMforward6, ITMforward10, ITMforward14, ITMforward18);
        System.out.printf(Locale.US, "%-18s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "Lognormal", ITMforward6Lognormal, ITMforward10Lognormal, ITMforward14Lognormal, ITMforward18Lognormal);
        System.out.println("-".repeat(70));
        System.out.println("");

        System.out.println("--- ATM Caplet (strike = 3 %) | Theoretical value: " + ATMtheoretical + " under Bachelier, " + ATMtheoreticalBlack + " under Black");
        System.out.println("-".repeat(70));
        System.out.printf(Locale.US, "%-18s  %12s  %12s  %12s  %12s%n", "State Space", "Forward 6", "Forward 10", "Forward 14", "Forward 18");
        System.out.println("-".repeat(70));
        System.out.printf(Locale.US, "%-18s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "Normal", ATMforward6, ATMforward10, ATMforward14, ATMforward18);
        System.out.printf(Locale.US, "%-18s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "Lognormal", ATMforward6Lognormal, ATMforward10Lognormal, ATMforward14Lognormal, ATMforward18Lognormal);
        System.out.println("-".repeat(70));
        System.out.println("");

        System.out.println("--- OTM Caplet (strike = 5 %) | Theoretical value: " + OTMtheoretical + " under Bachelier, " + OTMtheoreticalBlack + " under Black");
        System.out.println("-".repeat(70));
        System.out.printf(Locale.US, "%-18s  %12s  %12s  %12s  %12s%n", "State Space", "Forward 6", "Forward 10", "Forward 14", "Forward 18");
        System.out.println("-".repeat(70));
        System.out.printf(Locale.US, "%-18s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "Normal", OTMforward6, OTMforward10, OTMforward14, OTMforward18);
        System.out.printf(Locale.US, "%-18s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "Lognormal", OTMforward6Lognormal, OTMforward10Lognormal, OTMforward14Lognormal, OTMforward18Lognormal);
        System.out.println("-".repeat(70));
        System.out.println("");
    }

    // =========================================================================
	// Ex3 — Real-world measure P: ZeroDrift and RidingTheCurve
	// =========================================================================
	@Test
	public void ex3_realWorldMeasure() throws CalculationException {

		// Exercise-prescribed curve: L_0=2%, L_1=3%, L_j=4% for j>=2
		final int numberOfRates = (int)(TIME_HORIZON / PERIOD_LENGTH);
		final double[] fixingTimes = new double[numberOfRates];
		final double[] initialRates = new double[numberOfRates];
		for(int j = 0; j < numberOfRates; j++) {
			fixingTimes[j] = j * PERIOD_LENGTH;
			initialRates[j] = (j == 0) ? 0.02 : (j == 1) ? 0.03 : 0.04;
		}

		final ForwardCurve curve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"curve", fixingTimes, initialRates, PERIOD_LENGTH);

		// Build Model 1 (ZeroDrift)
		final LIBORMonteCarloSimulationFromLIBORModel simZero   = buildRealWorldSimulation(new ZeroDriftMeasure(), curve, SEED_ZERO_DRIFT);
        // Build Model 2 (RidingTheCurve)
		final LIBORMonteCarloSimulationFromLIBORModel simRiding = buildRealWorldSimulation(new RidingTheCurveMeasure(curve),  curve, SEED_RIDING);

		// Pairs (j, k) to check
		final int[][] pairs = { {1, 1}, {2, 1}, {2, 2}, {3, 1}, {3, 2}, {3, 3}};

		System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 3 — Real-World Measure P");
		System.out.println("  " + describeCurve(initialRates));
		System.out.println("=".repeat(80));
		System.out.println("");

		// --- Model 1: ZeroDrift — E[L_j(T_k)] = L_j(0)
		System.out.println("  " + "─".repeat(78));
		System.out.println("  Model 1: ZeroDriftMeasure — E[L_j(T_k)] = L_j(0)");
		System.out.println("  " + "─".repeat(78));
		System.out.printf(Locale.US, "  %-6s %-6s %12s %12s %12s%n", "j", "k", "L_j(0)", "E[L_j(T_k)]", "Abs. Error");
		System.out.println("  " + "-".repeat(70));
		for(final int[] pair : pairs) {
			final int j = pair[0];
			final int k = pair[1];
			final double periodLength = PERIOD_LENGTH;
			final double expected = curve.getForward(null, j * PERIOD_LENGTH, periodLength);
			final double simulated = simZero.getLIBOR(k, j).getAverage();
			System.out.printf(Locale.US, "  %-6d %-6d %+12.6f %+12.6f %+12.2e%n", j, k, expected, simulated, Math.abs(simulated - expected));
			Assertions.assertEquals(expected, simulated, 1e-3, "ZeroDrift: E[L_" + j + "(T_" + k + ")] = L_" + j + "(0)");
		}

		System.out.println("");
		// --- Model 2: RidingTheCurve — E[L_j(T_k)] = L_{j-k}(0)
		System.out.println("  " + "─".repeat(78));
		System.out.println("  Model 2: RidingTheCurveMeasure — E[L_j(T_k)] = L_{j-k}(0)");
		System.out.println("  " + "─".repeat(78));
		System.out.printf(Locale.US, "  %-6s %-6s %12s %12s %12s%n", "j", "k", "L_{j-k}(0)", "E[L_j(T_k)]", "Abs. Error");
		System.out.println("  " + "-".repeat(70));
		for(final int[] pair : pairs) {
			final int j = pair[0];
			final int k = pair[1];
			final double expected = curve.getForward(null, (j - k) * PERIOD_LENGTH, PERIOD_LENGTH);  // L_{j-k}(0)
			final double simulated = simRiding.getLIBOR(k, j).getAverage();
			System.out.printf(Locale.US, "  %-6d %-6d %+12.6f %+12.6f %+12.2e%n", j, k, expected, simulated, Math.abs(simulated - expected));
			Assertions.assertEquals(expected, simulated, 1e-2, "RidingTheCurve: E[L_" + j + "(T_" + k + ")] = L_" + (j-k) + "(0)");
		}
		System.out.println("");
	}

    // Ex5 — Displaced Log-Normal state-space transform
    // =========================================================================
    @Test
    public void ex5_displacedLognormal() throws CalculationException {
        final double fixingTime  = 4.5;
        final double paymentTime = 5.0;
        final double notional = 10000.0;
        final double strikeITM = 0.01;
        final double strikeOTM = 0.05;

        final int fixingTimeIndex = (int) (fixingTime / PERIOD_LENGTH);
        final int numberOfPeriods = (int) (TIME_HORIZON / PERIOD_LENGTH);

        final Caplet capletITM = new Caplet(fixingTime, paymentTime-fixingTime, strikeITM);
        final Caplet capletATM = new Caplet(fixingTime, paymentTime-fixingTime, FLAT_FORWARD_RATE);
        final Caplet capletOTM = new Caplet(fixingTime, paymentTime-fixingTime, strikeOTM);

        // ----- Section 4: Displaced log-normal state space transform
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginSpotDisplacedLognormal       = buildNewSimulation(new SpotMeasure(), new DisplacedLogNormalStateSpaceTransform(0.01, numberOfPeriods), SEED_SPOT);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginTerminalDisplacedLognormal   = buildNewSimulation(new TerminalMeasure(), new DisplacedLogNormalStateSpaceTransform(0.01, numberOfPeriods), SEED_TERMINAL);


        final double ITMpluginSpotDisplacedLognormal        = capletITM.getValue(simulationPluginSpotDisplacedLognormal) * notional;
        final double ATMpluginSpotDisplacedLognormal        = capletATM.getValue(simulationPluginSpotDisplacedLognormal) * notional;
        final double OTMpluginSpotDisplacedLognormal        = capletOTM.getValue(simulationPluginSpotDisplacedLognormal) * notional;

        final double ITMpluginTerminalDisplacedLognormal    = capletITM.getValue(simulationPluginTerminalDisplacedLognormal) * notional;
        final double ATMpluginTerminalDisplacedLognormal    = capletATM.getValue(simulationPluginTerminalDisplacedLognormal) * notional;
        final double OTMpluginTerminalDisplacedLognormal    = capletOTM.getValue(simulationPluginTerminalDisplacedLognormal) * notional;

        // get analytic values for comparison
        final double ITMtheoreticalDisplacedLognormal = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE + 0.01, strikeITM + 0.01, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double ATMtheoreticalDisplacedLognormal = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE + 0.01, FLAT_FORWARD_RATE + 0.01, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double OTMtheoreticalDisplacedLognormal = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE + 0.01, strikeOTM + 0.01, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

        System.out.println("=".repeat(80));
        System.out.println("  EXERCISE 5 — Displaced Log-Normal State-Space Transform");
        System.out.println("  Caplet: Fixing = 4.5 yr, Payment = 5.0 yr | Flat forward curve at 3% | Displacement = 1%");
        System.out.println("=".repeat(80));
        System.out.println("");

        System.out.println("--- ITM Caplet (strike = 1 %) | Theoretical value: " + ITMtheoreticalDisplacedLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s %12s%n", "Measure", "Numerical");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s %+12.6f%n", "Spot",     ITMpluginSpotDisplacedLognormal);
        System.out.printf(Locale.US, "  %-18s %+12.6f%n", "Terminal", ITMpluginTerminalDisplacedLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.println("");    

        System.out.println("--- ATM Caplet (strike = 3 %) | Theoretical value: " + ATMtheoreticalDisplacedLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s %12s%n", "Measure", "Numerical");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s %+12.6f%n", "Spot",     ATMpluginSpotDisplacedLognormal);    
        System.out.printf(Locale.US, "  %-18s %+12.6f%n", "Terminal", ATMpluginTerminalDisplacedLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.println("");

        System.out.println("--- OTM Caplet (strike = 5 %) | Theoretical value: " + OTMtheoreticalDisplacedLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s %12s%n", "Measure", "Numerical");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s %+12.6f%n", "Spot",     OTMpluginSpotDisplacedLognormal);    
        System.out.printf(Locale.US, "  %-18s %+12.6f%n", "Terminal", OTMpluginTerminalDisplacedLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.println("");

        Assertions.assertEquals(ITMpluginSpotDisplacedLognormal, ITMtheoreticalDisplacedLognormal, 1e-0, "ITM caplet consistency: Spot under displaced log-normal");
        Assertions.assertEquals(ITMpluginTerminalDisplacedLognormal, ITMtheoreticalDisplacedLognormal, 1e-0, "ITM caplet consistency: Terminal under displaced log-normal");
        Assertions.assertEquals(ATMpluginSpotDisplacedLognormal, ATMtheoreticalDisplacedLognormal, 1e-0, "ATM caplet consistency: Spot under displaced log-normal");
        Assertions.assertEquals(ATMpluginTerminalDisplacedLognormal, ATMtheoreticalDisplacedLognormal, 1e-0, "ATM caplet consistency: Terminal under displaced log-normal");
        Assertions.assertEquals(OTMpluginSpotDisplacedLognormal, OTMtheoreticalDisplacedLognormal, 1e-0, "OTM caplet consistency: Spot under displaced log-normal");
        Assertions.assertEquals(OTMpluginTerminalDisplacedLognormal, OTMtheoreticalDisplacedLognormal, 1e-0, "OTM caplet consistency: Terminal under displaced log-normal");

        // Section 5: Compare displaced log-normal state space transform with displacement = 0 with log-normal state space transforms
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginSpotLognormal       = buildNewSimulation(new SpotMeasure(), new LogNormalStateSpaceTransform(), SEED_SPOT);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginTerminalLognormal   = buildNewSimulation(new TerminalMeasure(), new LogNormalStateSpaceTransform(), SEED_TERMINAL);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginSpotDisplacedLognormalZero       = buildNewSimulation(new SpotMeasure(), new DisplacedLogNormalStateSpaceTransform(0.0, numberOfPeriods), SEED_SPOT);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginTerminalDisplacedLognormalZero   = buildNewSimulation(new TerminalMeasure(), new DisplacedLogNormalStateSpaceTransform(0.0, numberOfPeriods), SEED_TERMINAL);

        final double ITMpluginSpotLognormal        = capletITM.getValue(simulationPluginSpotLognormal) * notional;
        final double ATMpluginSpotLognormal        = capletATM.getValue(simulationPluginSpotLognormal) * notional;
        final double OTMpluginSpotLognormal        = capletOTM.getValue(simulationPluginSpotLognormal) * notional;
        
        final double ITMpluginTerminalLognormal    = capletITM.getValue(simulationPluginTerminalLognormal) * notional;
        final double ATMpluginTerminalLognormal    = capletATM.getValue(simulationPluginTerminalLognormal) * notional;
        final double OTMpluginTerminalLognormal    = capletOTM.getValue(simulationPluginTerminalLognormal) * notional;

        final double ITMpluginSpotDisplacedLognormalZero        = capletITM.getValue(simulationPluginSpotDisplacedLognormalZero) * notional;
        final double ATMpluginSpotDisplacedLognormalZero        = capletATM.getValue(simulationPluginSpotDisplacedLognormalZero) * notional;
        final double OTMpluginSpotDisplacedLognormalZero        = capletOTM.getValue(simulationPluginSpotDisplacedLognormalZero) * notional;

        final double ITMpluginTerminalDisplacedLognormalZero    = capletITM.getValue(simulationPluginTerminalDisplacedLognormalZero) * notional;
        final double ATMpluginTerminalDisplacedLognormalZero    = capletATM.getValue(simulationPluginTerminalDisplacedLognormalZero) * notional;
        final double OTMpluginTerminalDisplacedLognormalZero    = capletOTM.getValue(simulationPluginTerminalDisplacedLognormalZero) * notional;

        // get analytic values for comparison
        final double ITMtheoreticalLognormal = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeITM, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double ATMtheoreticalLognormal = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double OTMtheoreticalLognormal = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeOTM, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

        System.out.println("  " + "─".repeat(78));
        System.out.println("  Displaced LogNormal (d=0) vs. LogNormal — consistency check");
        System.out.println("  " + "─".repeat(78));
        System.out.println(""); 
        System.out.println("--- ITM Caplet (strike = 1 %) | Theoretical value: " + ITMtheoreticalLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "State Space", "Lognormal", "Displaced Lognormal (d=0)", "Abs. Difference");
        System.out.println("  ");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.6f%n", "Spot",     ITMpluginSpotLognormal, ITMpluginSpotDisplacedLognormalZero, Math.abs(ITMpluginSpotLognormal - ITMpluginSpotDisplacedLognormalZero));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.6f%n", "Terminal", ITMpluginTerminalLognormal, ITMpluginTerminalDisplacedLognormalZero, Math.abs(ITMpluginTerminalLognormal - ITMpluginTerminalDisplacedLognormalZero));
        System.out.println("  " + "-".repeat(70));
        System.out.println("");
        System.out.println("--- ATM Caplet (strike = 3 %) | Theoretical value: " + ATMtheoreticalLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "State Space", "Lognormal", "Displaced Lognormal (d=0)", "Abs. Difference");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.6f%n", "Spot",     ATMpluginSpotLognormal, ATMpluginSpotDisplacedLognormalZero, Math.abs(ATMpluginSpotLognormal - ATMpluginSpotDisplacedLognormalZero));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.6f%n", "Terminal", ATMpluginTerminalLognormal, ATMpluginTerminalDisplacedLognormalZero, Math.abs(ATMpluginTerminalLognormal - ATMpluginTerminalDisplacedLognormalZero));
        System.out.println("  " + "-".repeat(70));
        System.out.println("");
        System.out.println("--- OTM Caplet (strike = 5 %) | Theoretical value: " + OTMtheoreticalLognormal);
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "State Space", "Lognormal", "Displaced Lognormal (d=0)", "Abs. Difference");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.6f%n", "Spot",     OTMpluginSpotLognormal, OTMpluginSpotDisplacedLognormalZero, Math.abs(OTMpluginSpotLognormal - OTMpluginSpotDisplacedLognormalZero));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.6f%n", "Terminal", OTMpluginTerminalLognormal, OTMpluginTerminalDisplacedLognormalZero, Math.abs(OTMpluginTerminalLognormal - OTMpluginTerminalDisplacedLognormalZero));
        System.out.println("  " + "-".repeat(70));
        System.out.println(""); 

        Assertions.assertEquals(ITMpluginSpotLognormal, ITMpluginSpotDisplacedLognormalZero, 1e-10, "ITM caplet consistency: Spot under log-normal vs. displaced log-normal with zero displacement");
        Assertions.assertEquals(ITMpluginTerminalLognormal, ITMpluginTerminalDisplacedLognormalZero, 1e-10, "ITM caplet consistency: Terminal under log-normal vs. displaced log-normal with zero displacement");
        Assertions.assertEquals(ATMpluginSpotLognormal, ATMpluginSpotDisplacedLognormalZero, 1e-10, "ATM caplet consistency: Spot under log-normal vs. displaced log-normal with zero displacement");
        Assertions.assertEquals(ATMpluginTerminalLognormal, ATMpluginTerminalDisplacedLognormalZero, 1e-10, "ATM caplet consistency: Terminal under log-normal vs. displaced log-normal with zero displacement");
        Assertions.assertEquals(OTMpluginSpotLognormal, OTMpluginSpotDisplacedLognormalZero, 1e-10, "OTM caplet consistency: Spot under log-normal vs. displaced log-normal with zero displacement");
        Assertions.assertEquals(OTMpluginTerminalLognormal, OTMpluginTerminalDisplacedLognormalZero, 1e-10, "OTM caplet consistency: Terminal under log-normal vs. displaced log-normal with zero displacement");
    }


	// =========================================================================
	// Ex6 (+ Ex 2)— State-space transform plug-in: Normal + Log-Normal
	// =========================================================================
	@Test
	public void ex6_stateSpaceTransform() throws CalculationException {
		
        final double fixingTime  = 4.5;
		final double paymentTime = 5.0;
        final double notional = 10000.0;
        final double strikeITM = 0.01;
        final double strikeOTM = 0.05;

        final int fixingTimeIndex = (int) (fixingTime / PERIOD_LENGTH);

        final Caplet capletITM = new Caplet(fixingTime, paymentTime-fixingTime, strikeITM);
        final Caplet capletATM = new Caplet(fixingTime, paymentTime-fixingTime, FLAT_FORWARD_RATE);
        final Caplet capletOTM = new Caplet(fixingTime, paymentTime-fixingTime, strikeOTM);


		// ----- Section 1: Normal state space transform
		final LIBORMonteCarloSimulationFromLIBORModel simulationOriginalSpot     = buildOldSimulation("SPOT",     "NORMAL", SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel simulationOriginalTerminal = buildOldSimulation("TERMINAL", "NORMAL", SEED_TERMINAL);
		final LIBORMonteCarloSimulationFromLIBORModel simulationPluginSpot       = buildNewSimulation(new SpotMeasure(), new NormalStateSpaceTransform(), SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel simulationPluginTerminal   = buildNewSimulation(new TerminalMeasure(), new NormalStateSpaceTransform(), SEED_TERMINAL);

        // get numerical values
        final double ITMoriginalSpot      = capletITM.getValue(simulationOriginalSpot) * notional;
        final double ATMoriginalSpot      = capletATM.getValue(simulationOriginalSpot) * notional;
        final double OTMoriginalSpot      = capletOTM.getValue(simulationOriginalSpot) * notional;

        final double ITMoriginalTerminal  = capletITM.getValue(simulationOriginalTerminal) * notional;
        final double ATMoriginalTerminal  = capletATM.getValue(simulationOriginalTerminal) * notional;
        final double OTMoriginalTerminal  = capletOTM.getValue(simulationOriginalTerminal) * notional;

        final double ITMpluginSpot        = capletITM.getValue(simulationPluginSpot) * notional;
        final double ATMpluginSpot        = capletATM.getValue(simulationPluginSpot) * notional;
        final double OTMpluginSpot        = capletOTM.getValue(simulationPluginSpot) * notional;

        final double ITMpluginTerminal    = capletITM.getValue(simulationPluginTerminal) * notional;
        final double ATMpluginTerminal    = capletATM.getValue(simulationPluginTerminal) * notional;
        final double OTMpluginTerminal    = capletOTM.getValue(simulationPluginTerminal) * notional;

        // get analytic values for comparison
        final double normalizedVolatilityParameterA = FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A; // multiplied with initial forward rate for normal dynamics
        final double ITMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, strikeITM, fixingTimeIndex, normalizedVolatilityParameterA, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double ATMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingTimeIndex, normalizedVolatilityParameterA, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double OTMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, strikeOTM, fixingTimeIndex, normalizedVolatilityParameterA, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

        System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 6 (+ EXERCISE 2) — State-Space Transform Plug-In ( + Spot/Terminal Measure Plug-In)");
		System.out.println("  Caplet: Fixing = 4.5 yr, Payment = 5.0 yr | Flat forward curve at 3%");
		System.out.println("=".repeat(80));
		System.out.println("");
		System.out.println("  " + "─".repeat(78));
		System.out.println("  Normal (Bachelier) — bit-for-bit consistency with original model");
		System.out.println("  " + "─".repeat(78));

        // ITM caplet
		System.out.println("  --- ITM Caplet (strike = 1 %) | Theoretical value: " + ITMtheoretical);
		System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Abs. Error");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Spot",     ITMoriginalSpot,     ITMpluginSpot,     Math.abs(ITMoriginalSpot - ITMpluginSpot));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal", ITMoriginalTerminal, ITMpluginTerminal, Math.abs(ITMoriginalTerminal - ITMpluginTerminal));
		System.out.println("  " + "-".repeat(70));
        System.out.println("");

        // ATM caplet
        System.out.println("  --- ATM Caplet (strike = 3 %) | Theoretical value: " + ATMtheoretical);
        System.out.println("  " + "-".repeat(70));  
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Abs. Error");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Spot",     ATMoriginalSpot,     ATMpluginSpot,     Math.abs(ATMoriginalSpot - ATMpluginSpot));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal", ATMoriginalTerminal, ATMpluginTerminal, Math.abs(ATMoriginalTerminal - ATMpluginTerminal));
        System.out.println("  " + "-".repeat(70));
        System.out.println("");

        // OTM caplet
        System.out.println("  --- OTM Caplet (strike = 5 %) | Theoretical value: " + OTMtheoretical);
        System.out.println("  " + "-".repeat(70));  
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Abs. Error");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Spot",     OTMoriginalSpot,     OTMpluginSpot,     Math.abs(OTMoriginalSpot - OTMpluginSpot));  
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal", OTMoriginalTerminal, OTMpluginTerminal, Math.abs(OTMoriginalTerminal - OTMpluginTerminal));
        System.out.println("  " + "-".repeat(70));
        System.out.println("");

        Assertions.assertEquals(ITMoriginalSpot,     ITMpluginSpot,     1e-10, "ITM caplet consistency: Spot under Bachelier");
		Assertions.assertEquals(ITMoriginalTerminal, ITMpluginTerminal, 1e-10, "ITM caplet consistency: Terminal under Bachelier");
        Assertions.assertEquals(ATMoriginalSpot,     ATMpluginSpot,     1e-10, "ATM caplet consistency: Spot under Bachelier");
        Assertions.assertEquals(ATMoriginalTerminal, ATMpluginTerminal, 1e-10, "ATM caplet consistency: Terminal under Bachelier");
        Assertions.assertEquals(OTMoriginalSpot,     OTMpluginSpot,     1e-10, "OTM caplet consistency: Spot under Bachelier");
        Assertions.assertEquals(OTMoriginalTerminal, OTMpluginTerminal, 1e-10, "OTM caplet consistency: Terminal under Bachelier");

		System.out.println("");
		System.out.println("  " + "─".repeat(78));
		System.out.println("  Log-Normal (Black) — bit-for-bit consistency with original model");
		System.out.println("  " + "─".repeat(78));
		System.out.println("");

        final LIBORMonteCarloSimulationFromLIBORModel simulationOriginalSpotLognormal     = buildOldSimulation("SPOT",     "LOGNORMAL", SEED_SPOT);
        final LIBORMonteCarloSimulationFromLIBORModel simulationOriginalTerminalLognormal = buildOldSimulation("TERMINAL", "LOGNORMAL", SEED_TERMINAL);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginSpotLognormal       = buildNewSimulation(new SpotMeasure(), new LogNormalStateSpaceTransform(), SEED_SPOT);
        final LIBORMonteCarloSimulationFromLIBORModel simulationPluginTerminalLognormal   = buildNewSimulation(new TerminalMeasure(), new LogNormalStateSpaceTransform(), SEED_TERMINAL);

        final double ITMoriginalSpotLognormal      = capletITM.getValue(simulationOriginalSpotLognormal) * notional;
        final double ATMoriginalSpotLognormal      = capletATM.getValue(simulationOriginalSpotLognormal) * notional;
        final double OTMoriginalSpotLognormal      = capletOTM.getValue(simulationOriginalSpotLognormal) * notional;

        final double ITMoriginalTerminalLognormal  = capletITM.getValue(simulationOriginalTerminalLognormal) * notional;
        final double ATMoriginalTerminalLognormal  = capletATM.getValue(simulationOriginalTerminalLognormal) * notional;
        final double OTMoriginalTerminalLognormal  = capletOTM.getValue(simulationOriginalTerminalLognormal) * notional;

        final double ITMpluginSpotLognormal        = capletITM.getValue(simulationPluginSpotLognormal) * notional;
        final double ATMpluginSpotLognormal        = capletATM.getValue(simulationPluginSpotLognormal) * notional;
        final double OTMpluginSpotLognormal        = capletOTM.getValue(simulationPluginSpotLognormal) * notional;

        final double ITMpluginTerminalLognormal    = capletITM.getValue(simulationPluginTerminalLognormal) * notional;
        final double ATMpluginTerminalLognormal    = capletATM.getValue(simulationPluginTerminalLognormal) * notional;
        final double OTMpluginTerminalLognormal    = capletOTM.getValue(simulationPluginTerminalLognormal) * notional;

        // get analytic values for comparison
        final double ITMtheoreticalBlack = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeITM, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double ATMtheoreticalBlack = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
        final double OTMtheoreticalBlack = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeOTM, fixingTimeIndex, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

        // ITM caplet
		System.out.println("  --- ITM Caplet (strike = 1 %) | Theoretical value: " + ITMtheoreticalBlack);
		System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Abs. Error");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Spot",     ITMoriginalSpotLognormal,     ITMpluginSpotLognormal,     Math.abs(ITMoriginalSpotLognormal - ITMpluginSpotLognormal));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal", ITMoriginalTerminalLognormal, ITMpluginTerminalLognormal, Math.abs(ITMoriginalTerminalLognormal - ITMpluginTerminalLognormal));
		System.out.println("  " + "-".repeat(70));
        System.out.println("");

        // ATM caplet
        System.out.println("  --- ATM Caplet (strike = 3 %) | Theoretical value: " + ATMtheoreticalBlack);
        System.out.println("  " + "-".repeat(70));  
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Abs. Error");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Spot",     ATMoriginalSpotLognormal,     ATMpluginSpotLognormal,     Math.abs(ATMoriginalSpotLognormal - ATMpluginSpotLognormal));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal", ATMoriginalTerminalLognormal, ATMpluginTerminalLognormal, Math.abs(ATMoriginalTerminalLognormal - ATMpluginTerminalLognormal));
        System.out.println("  " + "-".repeat(70));
        System.out.println("");

        // OTM caplet
        System.out.println("  --- OTM Caplet (strike = 5 %) | Theoretical value: " + OTMtheoreticalBlack);
        System.out.println("  " + "-".repeat(70));  
        System.out.printf(Locale.US, "  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Abs. Error");
        System.out.println("  " + "-".repeat(70));
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Spot",     OTMoriginalSpotLognormal,     OTMpluginSpotLognormal,     Math.abs(OTMoriginalSpotLognormal - OTMpluginSpotLognormal));  
        System.out.printf(Locale.US, "  %-18s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal", OTMoriginalTerminalLognormal, OTMpluginTerminalLognormal, Math.abs(OTMoriginalTerminalLognormal - OTMpluginTerminalLognormal));
        System.out.println("  " + "-".repeat(70));
        System.out.println(""); 

        Assertions.assertEquals(ITMoriginalSpotLognormal,     ITMpluginSpotLognormal,     1e-10, "ITM caplet consistency: Spot under Black");
		Assertions.assertEquals(ITMoriginalTerminalLognormal, ITMpluginTerminalLognormal, 1e-10, "ITM caplet consistency: Terminal under Black");
        Assertions.assertEquals(ATMoriginalSpotLognormal,     ATMpluginSpotLognormal,     1e-10, "ATM caplet consistency: Spot under Black");
        Assertions.assertEquals(ATMoriginalTerminalLognormal, ATMpluginTerminalLognormal, 1e-10, "ATM caplet consistency: Terminal under Black");
        Assertions.assertEquals(OTMoriginalSpotLognormal,     OTMpluginSpotLognormal,     1e-10, "OTM caplet consistency: Spot under Black");
        Assertions.assertEquals(OTMoriginalTerminalLognormal, OTMpluginTerminalLognormal, 1e-10, "OTM caplet consistency: Terminal under Black");
    }

    // =========================================================================
	// -------------------------------------------------------------------------
	// Model construction helpers
	// -------------------------------------------------------------------------

	/**
	 * Builds a simulation using the original {@link LIBORMarketModelFromCovarianceModel}
	 * with an explicit state space.
	 *
	 * @param measureName    {@code "SPOT"} or {@code "TERMINAL"}.
	 * @param stateSpaceName {@code "NORMAL"} or {@code "LOGNORMAL"}.
	 * @param seed           Random seed for the Brownian motion.
	 */
	private LIBORMonteCarloSimulationFromLIBORModel buildOldSimulation(
			final String measureName, final String stateSpaceName, final int seed) throws CalculationException {

		final var timeDiscretization  = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);
		final var tenorDiscretization = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);

		final var forwardCurve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"forwardCurve",
				new double[]{ 0.5, TIME_HORIZON },
				new double[]{ FLAT_FORWARD_RATE, FLAT_FORWARD_RATE },
				PERIOD_LENGTH);
        
        // rescale volatility for different cases        
        final double volatilityParameterA = stateSpaceName.equalsIgnoreCase("NORMAL") ? FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A : VOLATILITY_PARAMETER_A;
        final var volatilityModel  = new LIBORVolatilityModelFourParameterExponentialForm(
                timeDiscretization, tenorDiscretization,
                volatilityParameterA, VOLATILITY_PARAMETER_B,
                VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, false);

		final var correlationModel = new LIBORCorrelationModelExponentialDecay(
				timeDiscretization, tenorDiscretization, NUMBER_OF_FACTORS, CORRELATION_DECAY);
		final var covarianceModel  = new LIBORCovarianceModelFromVolatilityAndCorrelation(
				timeDiscretization, tenorDiscretization, volatilityModel, correlationModel);

		final var properties = Map.of(
				"measure",    measureName,
				"stateSpace", stateSpaceName);

		final var model = new LIBORMarketModelFromCovarianceModel(
				tenorDiscretization, null, forwardCurve, null,
				new RandomVariableFromArrayFactory(), covarianceModel, null, properties);

		final var brownianMotion = new BrownianMotionFromMersenneRandomNumbers(
				timeDiscretization, NUMBER_OF_FACTORS, NUMBER_OF_PATHS, seed);
		final var process        = new EulerSchemeFromProcessModel(model, brownianMotion);

		return new LIBORMonteCarloSimulationFromLIBORModel(process);
	}

	/**
	 * Builds a simulation using our new {@link AugmentedLIBORMarketModel}
	 * with an explicit state-space transform.
	 *
	 * @param measure             The plug-in measure to inject.
	 * @param stateSpaceTransform The plug-in state-space transform to inject.
	 * @param seed                Random seed for the Brownian motion.
	 */
	private LIBORMonteCarloSimulationFromLIBORModel buildNewSimulation(
			final Measure measure, final StateSpaceTransform stateSpaceTransform, final int seed) throws CalculationException {

		final var timeDiscretization  = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);
		final var tenorDiscretization = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);

		final var forwardCurve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"forwardCurve",
				new double[]{ 0.5, TIME_HORIZON },
				new double[]{ FLAT_FORWARD_RATE, FLAT_FORWARD_RATE },
				PERIOD_LENGTH);

        // rescale volatility for different cases        
        final double volatilityParameterA = stateSpaceTransform.getClass().getSimpleName().equalsIgnoreCase("NormalStateSpaceTransform") ? FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A : VOLATILITY_PARAMETER_A;
        final var volatilityModel  = new LIBORVolatilityModelFourParameterExponentialForm(
                timeDiscretization, tenorDiscretization,
                volatilityParameterA, VOLATILITY_PARAMETER_B,
                VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, false);
		final var correlationModel = new LIBORCorrelationModelExponentialDecay(
				timeDiscretization, tenorDiscretization, NUMBER_OF_FACTORS, CORRELATION_DECAY);
		final var covarianceModel  = new LIBORCovarianceModelFromVolatilityAndCorrelation(
				timeDiscretization, tenorDiscretization, volatilityModel, correlationModel);

		final var model = new AugmentedLIBORMarketModel(
				tenorDiscretization, null, forwardCurve, null,
				new RandomVariableFromArrayFactory(), covarianceModel, measure, stateSpaceTransform, null, null);

		final var brownianMotion = new BrownianMotionFromMersenneRandomNumbers(
				timeDiscretization, NUMBER_OF_FACTORS, NUMBER_OF_PATHS, seed);
		final var process        = new EulerSchemeFromProcessModel(model, brownianMotion);

		return new LIBORMonteCarloSimulationFromLIBORModel(process);
	}

	/**
	 * Builds a real-world simulation (P-measure) using Normal state space.
	 *
	 * @param measure   The real-world measure plug-in (ZeroDrift or RidingTheCurve).
	 * @param curve     The initial forward curve.
	 * @param seed      Random seed.
	 */
	private LIBORMonteCarloSimulationFromLIBORModel buildRealWorldSimulation(
			final Measure measure, final ForwardCurve curve, final int seed) throws CalculationException {

		final var timeDiscretization  = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);
		final var tenorDiscretization = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);

		final double volatilityParameterA = FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A;
		final var volatilityModel  = new LIBORVolatilityModelFourParameterExponentialForm(
				timeDiscretization, tenorDiscretization,
				volatilityParameterA, VOLATILITY_PARAMETER_B,
				VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, false);
		final var correlationModel = new LIBORCorrelationModelExponentialDecay(
				timeDiscretization, tenorDiscretization, NUMBER_OF_FACTORS, CORRELATION_DECAY);
		final var covarianceModel  = new LIBORCovarianceModelFromVolatilityAndCorrelation(
				timeDiscretization, tenorDiscretization, volatilityModel, correlationModel);

		final var model = new AugmentedLIBORMarketModel(
				tenorDiscretization, null, curve, null,
				new RandomVariableFromArrayFactory(), covarianceModel, measure, new NormalStateSpaceTransform(), null, null);

		final var brownianMotion = new BrownianMotionFromMersenneRandomNumbers(
				timeDiscretization, NUMBER_OF_FACTORS, NUMBER_OF_PATHS, seed);
		final var process = new EulerSchemeFromProcessModel(model, brownianMotion);

		return new LIBORMonteCarloSimulationFromLIBORModel(process);
	}

	/**
	 * Returns a compact human-readable description of the initial forward curve,
	 * detecting where the rates become constant.
	 *
	 * @param rates Array of initial forward rates L_j(0).
	 * @return Compact curve description string.
	 */
	private static String describeCurve(final double[] rates) {
		final StringBuilder sb = new StringBuilder("Curve: ");
		for(int j = 0; j < rates.length; j++) {
			boolean constantFromHere = true;
			for(int k = j + 1; k < rates.length; k++) {
				if(rates[k] != rates[j]) { constantFromHere = false; break; }
			}
			if(j > 0) sb.append(", ");
			if(constantFromHere && j < rates.length - 1) {
				sb.append(String.format(Locale.US, "L_j(0)=%.2f%% for j>=%d", rates[j] * 100, j));
				break;
			}
			sb.append(String.format(Locale.US, "L_%d(0)=%.2f%%", j, rates[j] * 100));
		}
		return sb.toString();
	}

	// -------------------------------------------------------------------------
	// Product pricing helpers
	// -------------------------------------------------------------------------

	private double getAnalyticCapletValue(
		final String stateSpaceName,
		final double forwardRate,
		final double strike,
		final int LIBORIndex,
		final double volatilityParameterA,
        final double volatilityParameterB,
        final double volatilityParameterC,
        final double volatilityParameterD,
        final double notional
    ) {

        final double fixingTime = LIBORIndex * PERIOD_LENGTH;
        final double paymentTime = (LIBORIndex + 1) * PERIOD_LENGTH;
        final double volatility = getIntegratedVolatility(LIBORIndex, LIBORIndex, volatilityParameterA, volatilityParameterB, volatilityParameterC, volatilityParameterD);
        final double payoffUnit = PERIOD_LENGTH * 1.0 /Math.pow(1+FLAT_FORWARD_RATE*PERIOD_LENGTH, paymentTime/PERIOD_LENGTH);

        if (stateSpaceName.equalsIgnoreCase("LOGNORMAL")) {
			return AnalyticFormulas.blackScholesGeneralizedOptionValue(forwardRate, volatility, fixingTime, strike, payoffUnit) * notional;
		}
		if (stateSpaceName.equalsIgnoreCase("NORMAL")) {
			return AnalyticFormulas.bachelierOptionValue(forwardRate, volatility, fixingTime, strike, payoffUnit) * notional;
		}
		else {
			throw new IllegalArgumentException("Unknown state space: " + stateSpaceName);
		}
	}

    private double getIntegratedVolatility(
        final int LIBORIndex,
        final int timeIndex,
        final double volatilityParameterA,
        final double volatilityParameterB,
        final double volatilityParameterC,
        final double volatilityParameterD) {

        // See the documentation of LIBORVolatilityModelFourParameterExponentialForm for the formula of the integrated variance 
        double maturity = LIBORIndex * PERIOD_LENGTH;
        double variance = 0;
        for (int i = 0; i < timeIndex; i++) {
            double t = i * PERIOD_LENGTH;
            double tau = maturity - t;
            double sigma = (volatilityParameterA + volatilityParameterB * tau) * Math.exp(-volatilityParameterC * tau) + volatilityParameterD;
            variance += sigma * sigma * PERIOD_LENGTH;
        }
        variance /= timeIndex * PERIOD_LENGTH; // average variance per unit time
        return Math.sqrt(variance);
        }
}
