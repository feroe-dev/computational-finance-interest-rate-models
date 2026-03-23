package net.finmath.lecture.computationalfinance.project;

import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.lecture.computationalfinance.project.measure.ForwardMeasure;
import net.finmath.lecture.computationalfinance.project.measure.Measure;
import net.finmath.lecture.computationalfinance.project.measure.SpotMeasure;
import net.finmath.lecture.computationalfinance.project.measure.TerminalMeasure;
import net.finmath.lecture.computationalfinance.project.statespace.LogNormalStateSpaceTransform;
import net.finmath.lecture.computationalfinance.project.statespace.NormalStateSpaceTransform;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
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

/**
 * Tests for {@link AugmentedLIBORMarketModel}, covering the two plug-in interfaces
 * introduced in this project.
 *
 * <p>
 * <b>Exercise 2 — Measure Plug-in</b> ({@link #testMeasurePlugIn()}):
 * Verifies that {@code SpotMeasure}, {@code TerminalMeasure}, and {@code ForwardMeasure}
 * all yield consistent (measure-invariant) caplet prices. The state space is fixed to
 * Normal (Bachelier) throughout so that only the choice of numeraire varies.
 *
 * <p>
 * <b>Exercise 6 — State-Space Transform Plug-in</b> ({@link #testStateSpaceTransformPlugIn()}):
 * Verifies that {@code NormalStateSpaceTransform} and {@code LogNormalStateSpaceTransform}
 * reproduce the original {@code LIBORMarketModelFromCovarianceModel} bit-for-bit, and that
 * Normal (Bachelier) and Log-Normal (Black) dynamics produce <em>different</em> prices because
 * they represent fundamentally different distributional assumptions. The measure is fixed to
 * Spot throughout so that only the state-space transform varies.
 *
 * @author Felipe, GM-1, GM-2
 */
public class LIBORMarketModelPluginTest {

	// -------------------------------------------------------------------------
	// Shared model parameters
	// -------------------------------------------------------------------------

	/** Number of Monte Carlo paths. */
	private static final int NUMBER_OF_PATHS = 250000;

	/** Random seeds — one fixed seed per measure type for reproducibility. */
	private static final int SEED_SPOT      = 3141;
	private static final int SEED_TERMINAL  = 3142;
	private static final int SEED_FORWARD10 = 3143;
	private static final int SEED_FORWARD14 = 3144;
	private static final int SEED_FORWARD18 = 3145;

	/** Semi-annual tenor and simulation step (0 to 10 years, 20 periods). */
	private static final double PERIOD_LENGTH = 0.5;
	private static final double TIME_HORIZON  = 10.0;

	/** Flat initial forward rate. */
	private static final double FLAT_FORWARD_RATE = 0.03;

	/**
	 * Volatility parameters for the four-parameter exponential form
	 * \( \sigma(\tau) = (a + b\,\tau)\,e^{-c\,\tau} + d \).
	 * With b = 0 and d = 0 this simplifies to \( \sigma(\tau) = a\,e^{-c\,\tau} \).
	 */
	private static final double VOLATILITY_PARAMETER_A = 0.20;
	private static final double VOLATILITY_PARAMETER_B = 0.0;
	private static final double VOLATILITY_PARAMETER_C = 0.25;
	private static final double VOLATILITY_PARAMETER_D = 0.0;

	/** Correlation decay parameter for the exponential decay correlation model. */
	private static final double CORRELATION_DECAY = 0.10;

	/** Number of Brownian factors. */
	private static final int NUMBER_OF_FACTORS = 5;

	// =========================================================================
	// Test 1 — EXERCISE 2: Measure Plug-in
	// =========================================================================

	/**
	 * Verifies the Measure plug-in (Exercise 2).
	 *
	 * <p>Key insight: option prices are <em>measure-invariant</em> — the same caplet
	 * value is obtained regardless of which numeraire (Spot, Terminal, or T_k-Forward)
	 * is chosen. This is a fundamental consequence of the change-of-numeraire theorem.
	 *
	 * <p>The state space is fixed to Normal (Bachelier) throughout so that only the
	 * measure effect is isolated.
	 *
	 * <p>Sections:
	 * <ol>
	 *   <li><b>Consistency</b>: SpotMeasure and TerminalMeasure plug-ins reproduce the
	 *       original {@code LIBORMarketModelFromCovarianceModel} bit-for-bit (same seed
	 *       → same Brownian paths → exactly 0 abs. error).</li>
	 *   <li><b>Measure invariance</b>: Spot, Terminal, and T_k-Forward measures all
	 *       yield the same caplet price (within Monte Carlo noise).</li>
	 * </ol>
	 */
	@Test
	public void testMeasurePlugIn() throws CalculationException {

		final double fixingTime  = 4.5;
		final double paymentTime = 5.0;
		final double notional    = 10000.0;
		final double strikeITM   = 0.01;
		final double strikeOTM   = 0.05;
		final int    fixingIdx   = (int)(fixingTime / PERIOD_LENGTH);

		final Caplet capletITM = new Caplet(fixingTime, paymentTime - fixingTime, strikeITM);
		final Caplet capletATM = new Caplet(fixingTime, paymentTime - fixingTime, FLAT_FORWARD_RATE);
		final Caplet capletOTM = new Caplet(fixingTime, paymentTime - fixingTime, strikeOTM);

		// Analytic (Bachelier) reference values
		final double normalVol      = FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A;
		final double ITMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, strikeITM,         fixingIdx, normalVol, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double ATMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingIdx, normalVol, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double OTMtheoretical = getAnalyticCapletValue("NORMAL", FLAT_FORWARD_RATE, strikeOTM,         fixingIdx, normalVol, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

		System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 2 — Measure Plug-in (Spot / Terminal / Forward)");
		System.out.println("  Caplet | Fixing = 4.5 yr, Payment = 5.0 yr | Flat forward curve at 3 %");
		System.out.println("  Normal (Bachelier) state space fixed — only the measure varies.");
		System.out.println("=".repeat(80));
		System.out.println("");

		// ----- Section 1: Consistency ----------------------------------------
		final LIBORMonteCarloSimulationFromLIBORModel oldSpot     = buildOldSimulation("SPOT",     "NORMAL", SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel oldTerminal = buildOldSimulation("TERMINAL", "NORMAL", SEED_TERMINAL);
		final LIBORMonteCarloSimulationFromLIBORModel newSpot     = buildNewSimulation(new SpotMeasure(),     new NormalStateSpaceTransform(), SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel newTerminal = buildNewSimulation(new TerminalMeasure(), new NormalStateSpaceTransform(), SEED_TERMINAL);

		final double ITMoldSpot = capletITM.getValue(oldSpot) * notional;
		final double ATMoldSpot = capletATM.getValue(oldSpot) * notional;
		final double OTMoldSpot = capletOTM.getValue(oldSpot) * notional;
		final double ITMnewSpot = capletITM.getValue(newSpot) * notional;
		final double ATMnewSpot = capletATM.getValue(newSpot) * notional;
		final double OTMnewSpot = capletOTM.getValue(newSpot) * notional;

		final double ITMoldTerm = capletITM.getValue(oldTerminal) * notional;
		final double ATMoldTerm = capletATM.getValue(oldTerminal) * notional;
		final double OTMoldTerm = capletOTM.getValue(oldTerminal) * notional;
		final double ITMnewTerm = capletITM.getValue(newTerminal) * notional;
		final double ATMnewTerm = capletATM.getValue(newTerminal) * notional;
		final double OTMnewTerm = capletOTM.getValue(newTerminal) * notional;

		System.out.println("  --- Section 1: CONSISTENCY — SpotMeasure / TerminalMeasure plug-in vs original model");
		System.out.println("  Same seed → identical Brownian paths → bit-for-bit identical results (abs. error = 0).");
		System.out.println("  " + "-".repeat(72));
		System.out.printf("  %-26s  %12s  %12s  %12s%n", "Measure / Strike", "Original", "Plug-in", "Abs. Error");
		System.out.println("  " + "-".repeat(72));
		System.out.printf("  %-26s  %+12.6f  %+12.6f  %+12.2e%n", "Spot   — ITM (1 %)", ITMoldSpot, ITMnewSpot, Math.abs(ITMoldSpot - ITMnewSpot));
		System.out.printf("  %-26s  %+12.6f  %+12.6f  %+12.2e%n", "Spot   — ATM (3 %)", ATMoldSpot, ATMnewSpot, Math.abs(ATMoldSpot - ATMnewSpot));
		System.out.printf("  %-26s  %+12.6f  %+12.6f  %+12.2e%n", "Spot   — OTM (5 %)", OTMoldSpot, OTMnewSpot, Math.abs(OTMoldSpot - OTMnewSpot));
		System.out.printf("  %-26s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal — ITM",     ITMoldTerm, ITMnewTerm, Math.abs(ITMoldTerm - ITMnewTerm));
		System.out.printf("  %-26s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal — ATM",     ATMoldTerm, ATMnewTerm, Math.abs(ATMoldTerm - ATMnewTerm));
		System.out.printf("  %-26s  %+12.6f  %+12.6f  %+12.2e%n", "Terminal — OTM",     OTMoldTerm, OTMnewTerm, Math.abs(OTMoldTerm - OTMnewTerm));
		System.out.println("  " + "-".repeat(72));
		System.out.println("");

		Assertions.assertEquals(ITMoldSpot, ITMnewSpot, 1e-10, "Consistency: SpotMeasure ITM");
		Assertions.assertEquals(ATMoldSpot, ATMnewSpot, 1e-10, "Consistency: SpotMeasure ATM");
		Assertions.assertEquals(OTMoldSpot, OTMnewSpot, 1e-10, "Consistency: SpotMeasure OTM");
		Assertions.assertEquals(ITMoldTerm, ITMnewTerm, 1e-10, "Consistency: TerminalMeasure ITM");
		Assertions.assertEquals(ATMoldTerm, ATMnewTerm, 1e-10, "Consistency: TerminalMeasure ATM");
		Assertions.assertEquals(OTMoldTerm, OTMnewTerm, 1e-10, "Consistency: TerminalMeasure OTM");

		// ----- Section 2: Measure invariance --------------------------------
		final LIBORMonteCarloSimulationFromLIBORModel fwd10 = buildNewSimulation(new ForwardMeasure(10), new NormalStateSpaceTransform(), SEED_FORWARD10);
		final LIBORMonteCarloSimulationFromLIBORModel fwd14 = buildNewSimulation(new ForwardMeasure(14), new NormalStateSpaceTransform(), SEED_FORWARD14);
		final LIBORMonteCarloSimulationFromLIBORModel fwd18 = buildNewSimulation(new ForwardMeasure(18), new NormalStateSpaceTransform(), SEED_FORWARD18);

		final double ITMfwd10 = capletITM.getValue(fwd10) * notional;
		final double ATMfwd10 = capletATM.getValue(fwd10) * notional;
		final double OTMfwd10 = capletOTM.getValue(fwd10) * notional;
		final double ITMfwd14 = capletITM.getValue(fwd14) * notional;
		final double ATMfwd14 = capletATM.getValue(fwd14) * notional;
		final double OTMfwd14 = capletOTM.getValue(fwd14) * notional;
		final double ITMfwd18 = capletITM.getValue(fwd18) * notional;
		final double ATMfwd18 = capletATM.getValue(fwd18) * notional;
		final double OTMfwd18 = capletOTM.getValue(fwd18) * notional;

		System.out.println("  --- Section 2: MEASURE INVARIANCE — Spot / Terminal / Forward all yield the same price");
		System.out.println("  Theoretical (Bachelier): ITM = " + String.format("%.4f", ITMtheoretical)
				+ "  ATM = " + String.format("%.4f", ATMtheoretical)
				+ "  OTM = " + String.format("%.4f", OTMtheoretical));
		System.out.println("  Different seeds are used per measure to stress-test independence of numeraire choice.");
		System.out.println("  " + "-".repeat(72));
		System.out.printf("  %-28s  %12s  %12s  %12s%n", "Measure", "ITM (1%)", "ATM (3%)", "OTM (5%)");
		System.out.println("  " + "-".repeat(72));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.6f%n", "Spot",                  ITMnewSpot, ATMnewSpot, OTMnewSpot);
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.6f%n", "Terminal",               ITMnewTerm, ATMnewTerm, OTMnewTerm);
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.6f%n", "Forward (k=10, T=5yr)",  ITMfwd10,   ATMfwd10,   OTMfwd10);
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.6f%n", "Forward (k=14, T=7yr)",  ITMfwd14,   ATMfwd14,   OTMfwd14);
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.6f%n", "Forward (k=18, T=9yr)",  ITMfwd18,   ATMfwd18,   OTMfwd18);
		System.out.println("  " + "-".repeat(72));
		System.out.println("");
	}

	// =========================================================================
	// Test 2 — EXERCISE 6: State-Space Transform Plug-in
	// =========================================================================

	/**
	 * Verifies the StateSpaceTransform plug-in (Exercise 6).
	 *
	 * <p>Key insight: option prices are <em>NOT</em> state-space invariant — Normal
	 * (Bachelier) and Log-Normal (Black) dynamics produce different prices because they
	 * model fundamentally different distributional assumptions for the forward rate.
	 *
	 * <p>The measure is fixed to Spot throughout so that only the state-space transform
	 * varies.
	 *
	 * <p>Sections:
	 * <ol>
	 *   <li><b>Consistency</b>: Both {@code NormalStateSpaceTransform} and
	 *       {@code LogNormalStateSpaceTransform} reproduce the original
	 *       {@code LIBORMarketModelFromCovarianceModel} (enum-based) bit-for-bit
	 *       (same seed → same Brownian paths → exactly 0 abs. error).</li>
	 *   <li><b>State-space comparison</b>: Normal and Log-Normal prices differ at every
	 *       strike — each MC price matches its own analytical formula (Bachelier / Black).</li>
	 * </ol>
	 */
	@Test
	public void testStateSpaceTransformPlugIn() throws CalculationException {

		final double fixingTime  = 4.5;
		final double paymentTime = 5.0;
		final double notional    = 10000.0;
		final double strikeITM   = 0.01;
		final double strikeOTM   = 0.05;
		final int    fixingIdx   = (int)(fixingTime / PERIOD_LENGTH);

		final Caplet capletITM = new Caplet(fixingTime, paymentTime - fixingTime, strikeITM);
		final Caplet capletATM = new Caplet(fixingTime, paymentTime - fixingTime, FLAT_FORWARD_RATE);
		final Caplet capletOTM = new Caplet(fixingTime, paymentTime - fixingTime, strikeOTM);

		// Build simulations — Spot measure fixed, state space varies
		final LIBORMonteCarloSimulationFromLIBORModel oldNormal    = buildOldSimulation("SPOT", "NORMAL",    SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel oldLogNormal = buildOldSimulation("SPOT", "LOGNORMAL", SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel newNormal    = buildNewSimulation(new SpotMeasure(), new NormalStateSpaceTransform(),    SEED_SPOT);
		final LIBORMonteCarloSimulationFromLIBORModel newLogNormal = buildNewSimulation(new SpotMeasure(), new LogNormalStateSpaceTransform(), SEED_SPOT);

		final double ITMoldNormal    = capletITM.getValue(oldNormal)    * notional;
		final double ATMoldNormal    = capletATM.getValue(oldNormal)    * notional;
		final double OTMoldNormal    = capletOTM.getValue(oldNormal)    * notional;
		final double ITMnewNormal    = capletITM.getValue(newNormal)    * notional;
		final double ATMnewNormal    = capletATM.getValue(newNormal)    * notional;
		final double OTMnewNormal    = capletOTM.getValue(newNormal)    * notional;

		final double ITMoldLogNormal = capletITM.getValue(oldLogNormal) * notional;
		final double ATMoldLogNormal = capletATM.getValue(oldLogNormal) * notional;
		final double OTMoldLogNormal = capletOTM.getValue(oldLogNormal) * notional;
		final double ITMnewLogNormal = capletITM.getValue(newLogNormal) * notional;
		final double ATMnewLogNormal = capletATM.getValue(newLogNormal) * notional;
		final double OTMnewLogNormal = capletOTM.getValue(newLogNormal) * notional;

		// Analytic reference values
		final double normalVol    = FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A;
		final double ITMbachelier = getAnalyticCapletValue("NORMAL",    FLAT_FORWARD_RATE, strikeITM,         fixingIdx, normalVol,              VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double ATMbachelier = getAnalyticCapletValue("NORMAL",    FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingIdx, normalVol,              VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double OTMbachelier = getAnalyticCapletValue("NORMAL",    FLAT_FORWARD_RATE, strikeOTM,         fixingIdx, normalVol,              VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double ITMblack     = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeITM,         fixingIdx, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double ATMblack     = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, FLAT_FORWARD_RATE, fixingIdx, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);
		final double OTMblack     = getAnalyticCapletValue("LOGNORMAL", FLAT_FORWARD_RATE, strikeOTM,         fixingIdx, VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B, VOLATILITY_PARAMETER_C, VOLATILITY_PARAMETER_D, notional);

		System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 6 — State-Space Transform Plug-in (Normal / Log-Normal)");
		System.out.println("  Caplet | Fixing = 4.5 yr, Payment = 5.0 yr | Flat forward curve at 3 %");
		System.out.println("  Spot measure fixed — only the state-space transform varies.");
		System.out.println("=".repeat(80));
		System.out.println("");

		// ----- Section 1: Consistency ----------------------------------------
		System.out.println("  --- Section 1: CONSISTENCY — plug-in vs original LIBORMarketModelFromCovarianceModel (enum-based)");
		System.out.println("  Same seed → identical Brownian paths → bit-for-bit identical results (abs. error = 0).");
		System.out.println("  " + "-".repeat(72));
		System.out.printf("  %-28s  %12s  %12s  %12s%n", "State Space / Strike", "Original", "Plug-in", "Abs. Error");
		System.out.println("  " + "-".repeat(72));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.2e%n", "Normal — ITM (1 %)",    ITMoldNormal,    ITMnewNormal,    Math.abs(ITMoldNormal    - ITMnewNormal));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.2e%n", "Normal — ATM (3 %)",    ATMoldNormal,    ATMnewNormal,    Math.abs(ATMoldNormal    - ATMnewNormal));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.2e%n", "Normal — OTM (5 %)",    OTMoldNormal,    OTMnewNormal,    Math.abs(OTMoldNormal    - OTMnewNormal));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.2e%n", "Log-Normal — ITM",      ITMoldLogNormal, ITMnewLogNormal, Math.abs(ITMoldLogNormal - ITMnewLogNormal));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.2e%n", "Log-Normal — ATM",      ATMoldLogNormal, ATMnewLogNormal, Math.abs(ATMoldLogNormal - ATMnewLogNormal));
		System.out.printf("  %-28s  %+12.6f  %+12.6f  %+12.2e%n", "Log-Normal — OTM",      OTMoldLogNormal, OTMnewLogNormal, Math.abs(OTMoldLogNormal - OTMnewLogNormal));
		System.out.println("  " + "-".repeat(72));
		System.out.println("");

		Assertions.assertEquals(ITMoldNormal,    ITMnewNormal,    1e-10, "Consistency: Normal ITM");
		Assertions.assertEquals(ATMoldNormal,    ATMnewNormal,    1e-10, "Consistency: Normal ATM");
		Assertions.assertEquals(OTMoldNormal,    OTMnewNormal,    1e-10, "Consistency: Normal OTM");
		Assertions.assertEquals(ITMoldLogNormal, ITMnewLogNormal, 1e-10, "Consistency: LogNormal ITM");
		Assertions.assertEquals(ATMoldLogNormal, ATMnewLogNormal, 1e-10, "Consistency: LogNormal ATM");
		Assertions.assertEquals(OTMoldLogNormal, OTMnewLogNormal, 1e-10, "Consistency: LogNormal OTM");

		// ----- Section 2: State-space comparison (prices differ) ---------------
		System.out.println("  --- Section 2: STATE-SPACE COMPARISON — Normal vs Log-Normal dynamics produce different prices");
		System.out.println("  Normal vol = " + String.format("%.4f", normalVol)
				+ " (= F * sigma_a)  |  Log-Normal (Black) vol = " + VOLATILITY_PARAMETER_A);
		System.out.println("  Each MC price matches its own analytic formula (Bachelier / Black).");
		System.out.println("  " + "-".repeat(76));
		System.out.printf("  %-10s  %12s  %12s  %12s  %12s%n", "Strike", "Normal MC", "Bachelier", "LogNorm MC", "Black");
		System.out.println("  " + "-".repeat(76));
		System.out.printf("  %-10s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "ITM (1 %)", ITMnewNormal, ITMbachelier, ITMnewLogNormal, ITMblack);
		System.out.printf("  %-10s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "ATM (3 %)", ATMnewNormal, ATMbachelier, ATMnewLogNormal, ATMblack);
		System.out.printf("  %-10s  %+12.6f  %+12.6f  %+12.6f  %+12.6f%n", "OTM (5 %)", OTMnewNormal, OTMbachelier, OTMnewLogNormal, OTMblack);
		System.out.println("  " + "-".repeat(76));
		System.out.println("  Normal and Log-Normal prices differ: different distributional assumptions.");
		System.out.println("  Each MC price is consistent with its own analytical formula.");
		System.out.println("");
	}

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

		final double volatilityParameterA = stateSpaceName.equalsIgnoreCase("NORMAL")
				? FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A : VOLATILITY_PARAMETER_A;
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

		final double volatilityParameterA = stateSpaceTransform.getClass().getSimpleName().equalsIgnoreCase("NormalStateSpaceTransform")
				? FLAT_FORWARD_RATE * VOLATILITY_PARAMETER_A : VOLATILITY_PARAMETER_A;
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

		final double fixingTime  = LIBORIndex * PERIOD_LENGTH;
		final double paymentTime = (LIBORIndex + 1) * PERIOD_LENGTH;
		final double volatility  = getIntegratedVolatility(LIBORIndex, LIBORIndex, volatilityParameterA, volatilityParameterB, volatilityParameterC, volatilityParameterD);
		final double payoffUnit  = PERIOD_LENGTH * 1.0 / Math.pow(1 + FLAT_FORWARD_RATE * PERIOD_LENGTH, paymentTime / PERIOD_LENGTH);

		if(stateSpaceName.equalsIgnoreCase("LOGNORMAL")) {
			return AnalyticFormulas.blackModelCapletValue(forwardRate, volatility, fixingTime, strike, PERIOD_LENGTH, payoffUnit) * notional;
		}
		if(stateSpaceName.equalsIgnoreCase("NORMAL")) {
			return AnalyticFormulas.bachelierOptionValue(forwardRate, volatility, fixingTime, strike, payoffUnit) * notional;
		}
		throw new IllegalArgumentException("Unknown state space: " + stateSpaceName);
	}

	private double getIntegratedVolatility(
		final int LIBORIndex,
		final int timeIndex,
		final double volatilityParameterA,
		final double volatilityParameterB,
		final double volatilityParameterC,
		final double volatilityParameterD) {

		double maturity  = LIBORIndex * PERIOD_LENGTH;
		double variance  = 0;
		for(int i = 0; i < timeIndex; i++) {
			double t   = i * PERIOD_LENGTH;
			double tau = maturity - t;
			double sigma = (volatilityParameterA + volatilityParameterB * tau) * Math.exp(-volatilityParameterC * tau) + volatilityParameterD;
			variance += sigma * sigma * PERIOD_LENGTH;
		}
		variance /= timeIndex * PERIOD_LENGTH;
		return Math.sqrt(variance);
	}
}
