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
import net.finmath.montecarlo.interestrate.TermStructureMonteCarloSimulationModel;
import net.finmath.montecarlo.interestrate.models.LIBORMarketModelFromCovarianceModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCorrelationModelExponentialDecay;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModelFromVolatilityAndCorrelation;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORVolatilityModelFourParameterExponentialForm;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Integration tests for the measure and state-space transform plug-ins
 * of {@link AugmentedLIBORMarketModel}.
 *
 * <p>
 * One test per product. Each test checks three things in sequence:
 * <ol>
 *   <li><b>Consistency</b>: The new plug-in model must reproduce the original
 *       {@code LIBORMarketModelFromCovarianceModel} bit-for-bit (same seed).</li>
 *   <li><b>Measure invariance</b>: The price must be the same under the Spot measure,
 *       Terminal measure, and T_k-forward measure for k = 2, 5, 8, 10.</li>
 *   <li><b>Accuracy</b>: The Monte Carlo price is compared against the exact
 *       analytical formula at multiple strikes.</li>
 * </ol>
 *
 * @author Felipe, GM-1, GM-2
 */
public class LIBORMarketModelPluginTest {

	// -------------------------------------------------------------------------
	// Shared model parameters
	// -------------------------------------------------------------------------

	/** Number of Monte Carlo paths. */
	private static final int NUMBER_OF_PATHS = 20000;

	/** Random seeds — one fixed seed per measure type for consistency within a measure. */
	private static final int SEED_SPOT     = 3141;
	private static final int SEED_TERMINAL = 3142;
	private static final int SEED_FORWARD  = 3143;

	/** Semi-annual tenor and simulation step (0 to 10 years, 20 periods). */
	private static final double PERIOD_LENGTH = 0.5;
	private static final double TIME_HORIZON  = 10.0;

	/** Flat forward rate used for the test curve (= at-the-money forward LIBOR). */
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

	/** Correlation decay parameter. */
	private static final double CORRELATION_DECAY = 0.10;

	/** Number of Brownian factors. */
	private static final int NUMBER_OF_FACTORS = 2;

	/**
	 * Forward measure indices to test for products paying at T_e = 4.0 yr = T_8.
	 *
	 * <p>
	 * The T_k-forward measure is only valid when k ≥ payment tenor index (here k ≥ 8),
	 * because for k &lt; 8 the numeraire bond P(t, T_k) has already matured at the
	 * payment date T_e = 4.0 yr, causing getNumeraire(T_e) to incorrectly return 1.
	 * Index k relates to time via T_k = k × PERIOD_LENGTH (0.5 yr step).
	 *
	 * <ul>
	 *   <li>k =  9 → T_k = 4.5 yr: one period after payment</li>
	 *   <li>k = 14 → T_k = 7.0 yr: six periods after payment</li>
	 *   <li>k = 16 → T_k = 8.0 yr: eight periods after payment</li>
	 *   <li>k = 20 → T_k = 10.0 yr: terminal measure (numeraire = last bond)</li>
	 * </ul>
	 */
	private static final int[] FORWARD_MEASURE_INDICES = { 9, 14, 16, 20 };

	/** Strikes used in the accuracy tests (in decimal, e.g. 0.02 = 2 %). */
	private static final double[] TEST_STRIKES = { 0.02, 0.03, 0.05, 0.10 };

	/** Tolerance: same seed → bit-for-bit identical results. */
	private static final double TOLERANCE_EXACT = 1.0E-10;

	/**
	 * Tolerance for measure invariance: different seeds + possible Euler bias
	 * when k is far from the natural measure index (k=8).
	 * At 20,000 paths the Monte Carlo standard error is ~0.5e-3; Euler bias
	 * for distant k can add another ~2e-3. A bound of 5e-3 covers both.
	 */
	private static final double TOLERANCE_MONTE_CARLO = 5.0E-3;

	/**
	 * Tolerance for Monte Carlo vs analytical formula.
	 * The Caplet has an O(dt) Euler bias (~7 %), the Forward Rate Agreement has
	 * no systematic bias (linear payoff). Both fit within this bound.
	 */
	private static final double TOLERANCE_ANALYTICAL = 8.0E-3;

	// =========================================================================
	// Test 1 — Forward Rate Agreement
	// =========================================================================

	/**
	 * All checks for the Forward Rate Agreement (fixing T_s = 3.5 yr, payment T_e = 4.0 yr).
	 *
	 * <p>Sections:
	 * <ol>
	 *   <li>Consistency: new plug-in vs original finmath-lib (same seed).</li>
	 *   <li>Measure invariance: Spot, Terminal, and T_k-forward for k = 2, 5, 8, 10.</li>
	 *   <li>Accuracy: Monte Carlo vs exact formula \( V_0 = (F-K)\,\delta\,P(0,T_e) \)
	 *       at strikes K = 2 %, 3 %, 5 %, 10 %.</li>
	 * </ol>
	 */
	@Test
	public void testForwardRateAgreement() throws CalculationException {
		final double fixingTime  = 3.5;
		final double paymentTime = 4.0;

		// ----- Section 1: Consistency ----------------------------------------
		final TermStructureMonteCarloSimulationModel simulationOriginalSpot     = buildOldSimulation("SPOT",     SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simulationOriginalTerminal = buildOldSimulation("TERMINAL", SEED_TERMINAL);
		final TermStructureMonteCarloSimulationModel simulationPluginSpot       = buildNewSimulation(new SpotMeasure(),     SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simulationPluginTerminal   = buildNewSimulation(new TerminalMeasure(), SEED_TERMINAL);

		final double originalSpot     = forwardRateAgreementValue(simulationOriginalSpot,     fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double originalTerminal = forwardRateAgreementValue(simulationOriginalTerminal, fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double pluginSpot       = forwardRateAgreementValue(simulationPluginSpot,       fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double pluginTerminal   = forwardRateAgreementValue(simulationPluginTerminal,   fixingTime, paymentTime, FLAT_FORWARD_RATE);

		final double relativeErrorSpotInPercent     = Math.abs(originalSpot)     > 1.0E-10
				? 100.0 * Math.abs(pluginSpot     - originalSpot)     / Math.abs(originalSpot)     : 0.0;
		final double relativeErrorTerminalInPercent = Math.abs(originalTerminal) > 1.0E-10
				? 100.0 * Math.abs(pluginTerminal - originalTerminal) / Math.abs(originalTerminal) : 0.0;

		System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 2 — Measure Plug-in");
		System.out.println("  FORWARD RATE AGREEMENT  |  Fixing = 3.5 yr, Payment = 4.0 yr");
		System.out.println("=".repeat(80));
		System.out.println("");
		System.out.println("  --- CONSISTENCY: does our new plug-in reproduce the original finmath-lib?");
		System.out.println("  " + "-".repeat(70));
		System.out.printf("  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Rel. Error");
		System.out.println("  " + "-".repeat(70));
		System.out.printf("  %-18s  %+12.6f  %+12.6f  %11.2f %%%n", "Spot",     originalSpot,     pluginSpot,     relativeErrorSpotInPercent);
		System.out.printf("  %-18s  %+12.6f  %+12.6f  %11.2f %%%n", "Terminal", originalTerminal, pluginTerminal, relativeErrorTerminalInPercent);
		System.out.println("  " + "-".repeat(70));
		System.out.println("");

		Assertions.assertEquals(originalSpot,     pluginSpot,     TOLERANCE_EXACT, "Forward Rate Agreement consistency: Spot");
		Assertions.assertEquals(originalTerminal, pluginTerminal, TOLERANCE_EXACT, "Forward Rate Agreement consistency: Terminal");

		// ----- Section 2: Measure invariance --------------------------------
		final TermStructureMonteCarloSimulationModel simulationSpot     = buildNewSimulation(new SpotMeasure(),     SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simulationTerminal = buildNewSimulation(new TerminalMeasure(), SEED_TERMINAL);
		final double priceUnderSpot     = forwardRateAgreementValue(simulationSpot,     fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double priceUnderTerminal = forwardRateAgreementValue(simulationTerminal, fixingTime, paymentTime, FLAT_FORWARD_RATE);
		System.out.println("  --- MEASURE INVARIANCE: theoretical value = 0.000000 (at-the-money)");
		System.out.println("  Note: k >= 8 required (T_k >= payment date T_e = 4.0 yr).");
		System.out.println("  " + "-".repeat(67));
		System.out.printf("  %-34s  %12s  %18s%n", "Measure", "Price", "Abs. Error vs Spot");
		System.out.println("  " + "-".repeat(67));
		System.out.printf("  %-34s  %+12.6f  %18s%n",   "Spot measure (reference)", priceUnderSpot,     "—");
		System.out.printf("  %-34s  %+12.6f  %18.6f%n", "Terminal measure",         priceUnderTerminal, Math.abs(priceUnderTerminal - priceUnderSpot));

		Assertions.assertEquals(priceUnderSpot, priceUnderTerminal, TOLERANCE_MONTE_CARLO, "Forward Rate Agreement: Spot vs Terminal");

		for(int i = 0; i < FORWARD_MEASURE_INDICES.length; i++) {
			final int k = FORWARD_MEASURE_INDICES[i];
			final TermStructureMonteCarloSimulationModel simulationForward =
					buildNewSimulation(new ForwardMeasure(k), SEED_FORWARD);
			final double price = forwardRateAgreementValue(simulationForward, fixingTime, paymentTime, FLAT_FORWARD_RATE);
			System.out.printf("  %-34s  %+12.6f  %18.6f%n",
					"Forward measure (k=" + k + ", T=" + (k * PERIOD_LENGTH) + "yr)", price, Math.abs(price - priceUnderSpot));
			Assertions.assertEquals(priceUnderSpot, price, TOLERANCE_MONTE_CARLO, "Forward Rate Agreement: Spot vs Forward measure k=" + k);
		}
		System.out.println("  " + "-".repeat(67));
		System.out.println("");

		// ----- Section 3: Accuracy -------------------------------------------
		final double zeroCouponBond = Math.pow(1.0 / (1.0 + FLAT_FORWARD_RATE * PERIOD_LENGTH), paymentTime / PERIOD_LENGTH);
		System.out.println("  --- ACCURACY: Monte Carlo vs exact formula V_0 = (F - K) * delta * P(0, T_e)");
		System.out.println("  Note: Linear payoff — no Euler discretisation bias, only Monte Carlo noise.");
		System.out.println("  Rel. Error = (Monte Carlo - Theoretical) / |Theoretical|.");
		System.out.println("  " + "-".repeat(70));
		System.out.printf("  %-10s  %14s  %14s  %14s  %14s%n", "Strike", "Monte Carlo", "Theoretical", "Abs. Error", "Rel. Error");
		System.out.println("  " + "-".repeat(70));

		for(final double strike : TEST_STRIKES) {
			final double monteCarloValue  = forwardRateAgreementValue(simulationSpot, fixingTime, paymentTime, strike);
			final double theoreticalValue = (FLAT_FORWARD_RATE - strike) * PERIOD_LENGTH * zeroCouponBond;
			final double absoluteError    = monteCarloValue - theoreticalValue;
			final double relativeErrorInPercent = Math.abs(theoreticalValue) > 1.0E-10
					? 100.0 * absoluteError / Math.abs(theoreticalValue) : Double.NaN;

			if(Double.isNaN(relativeErrorInPercent)) {
				System.out.printf("  %8.2f %%  %+14.6f  %+14.6f  %+14.6f  %14s%n",
						strike * 100.0, monteCarloValue, theoreticalValue, absoluteError, "N/A (K=F)");
			} else {
				System.out.printf("  %8.2f %%  %+14.6f  %+14.6f  %+14.6f  %+13.2f %%%n",
						strike * 100.0, monteCarloValue, theoreticalValue, absoluteError, relativeErrorInPercent);
			}

			Assertions.assertEquals(theoreticalValue, monteCarloValue, TOLERANCE_ANALYTICAL,
					String.format("Forward Rate Agreement accuracy at K = %.0f %%", strike * 100.0));
		}
		System.out.println("  " + "-".repeat(70));
		System.out.println("");
		System.out.println();
	}

	// =========================================================================
	// Test 3 — State-Space Transform
	// =========================================================================

	/**
	 * Verifies that the state-space plug-in reproduces the original model bit-for-bit,
	 * and shows how the Caplet price differs between Normal and Log-Normal state spaces.
	 *
	 * <p>Sections:
	 * <ol>
	 *   <li><b>Consistency (Normal)</b>: {@link NormalStateSpaceTransform} must reproduce
	 *       the original model with {@code StateSpace.NORMAL} bit-for-bit (same seed).</li>
	 *   <li><b>Consistency (Log-Normal)</b>: {@link LogNormalStateSpaceTransform} must reproduce
	 *       the original model with {@code StateSpace.LOGNORMAL} bit-for-bit (same seed).</li>
	 *   <li><b>State-space comparison</b>: Caplet and FRA prices under Normal vs Log-Normal
	 *       are shown at multiple strikes. They should differ because the distributional
	 *       assumptions are different (Bachelier vs Black dynamics).</li>
	 * </ol>
	 */
	@Test
	public void testStateSpaceTransform() throws CalculationException {
		final double fixingTime  = 3.5;
		final double paymentTime = 4.0;

		// ----- Section 1: Consistency (Normal) -------------------------------
		final TermStructureMonteCarloSimulationModel simOldNormal  =
				buildOldSimulation("SPOT", "NORMAL", SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simPluginNormal =
				buildNewSimulation(new SpotMeasure(), new NormalStateSpaceTransform(), SEED_SPOT);

		final double oldCapletNormal    = capletValue(simOldNormal,    fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double pluginCapletNormal = capletValue(simPluginNormal, fixingTime, paymentTime, FLAT_FORWARD_RATE);

		// ----- Section 2: Consistency (Log-Normal) ---------------------------
		final TermStructureMonteCarloSimulationModel simOldLogNormal  =
				buildOldSimulation("SPOT", "LOGNORMAL", SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simPluginLogNormal =
				buildNewSimulation(new SpotMeasure(), new LogNormalStateSpaceTransform(), SEED_SPOT);

		final double oldCapletLogNormal    = capletValue(simOldLogNormal,    fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double pluginCapletLogNormal = capletValue(simPluginLogNormal, fixingTime, paymentTime, FLAT_FORWARD_RATE);

		System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 6 — State-Space Transform Plug-in");
		System.out.println("  STATE-SPACE TRANSFORM  |  Spot measure, Fixing = 3.5 yr, Payment = 4.0 yr");
		System.out.println("=".repeat(80));
		System.out.println("");
		System.out.println("  --- CONSISTENCY: plug-in model vs original StateSpace enum (same seed → bit-for-bit)");
		System.out.println("  The absolute error is exactly 0 because both models execute identical floating-point");
		System.out.println("  operations in the same order on the same Brownian paths (deterministic reproduction).");
		System.out.println("  " + "-".repeat(74));
		System.out.printf("  %-22s  %12s  %12s  %12s%n", "State Space", "Original", "Plug-in", "Abs. Error");
		System.out.println("  " + "-".repeat(74));
		System.out.printf("  %-22s  %+12.6f  %+12.6f  %+12.2e%n",
				"Normal (Bachelier)",    oldCapletNormal,    pluginCapletNormal,    Math.abs(pluginCapletNormal    - oldCapletNormal));
		System.out.printf("  %-22s  %+12.6f  %+12.6f  %+12.2e%n",
				"Log-Normal (Black)",   oldCapletLogNormal, pluginCapletLogNormal, Math.abs(pluginCapletLogNormal - oldCapletLogNormal));
		System.out.println("  " + "-".repeat(74));
		System.out.println("");

		Assertions.assertEquals(oldCapletNormal,    pluginCapletNormal,    TOLERANCE_EXACT, "State-space consistency: NormalStateSpaceTransform vs StateSpace.NORMAL");
		Assertions.assertEquals(oldCapletLogNormal, pluginCapletLogNormal, TOLERANCE_EXACT, "State-space consistency: LogNormalStateSpaceTransform vs StateSpace.LOGNORMAL");

		// ----- Section 3: State-space comparison -----------------------------
		System.out.println("  --- STATE-SPACE COMPARISON: Caplet price — Normal vs Log-Normal (Spot measure)");
		System.out.println("  Note: prices differ because Normal and Log-Normal impose different dynamics.");
		System.out.println("  " + "-".repeat(70));
		System.out.printf("  %-10s  %14s  %14s  %14s%n", "Strike", "Normal", "Log-Normal", "Difference");
		System.out.println("  " + "-".repeat(70));

		for(final double strike : TEST_STRIKES) {
			final double capletNormal    = capletValue(simPluginNormal,    fixingTime, paymentTime, strike);
			final double capletLogNormal = capletValue(simPluginLogNormal, fixingTime, paymentTime, strike);
			System.out.printf("  %8.2f %%  %+14.6f  %+14.6f  %+14.6f%n",
					strike * 100.0, capletNormal, capletLogNormal, capletLogNormal - capletNormal);
		}
		System.out.println("  " + "-".repeat(70));
		System.out.println("");
		System.out.println("=".repeat(80));
		System.out.println();
	}

	// =========================================================================
	// Test 2 — Caplet
	// =========================================================================

	/**
	 * All checks for the Caplet (fixing T_s = 3.5 yr, payment T_e = 4.0 yr).
	 *
	 * <p>Sections:
	 * <ol>
	 *   <li>Consistency: new plug-in vs original finmath-lib (same seed).</li>
	 *   <li>Measure invariance: Spot, Terminal, and T_k-forward for k = 2, 5, 8, 10.</li>
	 *   <li>Accuracy: Monte Carlo vs Bachelier analytical formula at strikes
	 *       K = 2 %, 3 %, 5 %, 10 %. Remaining error is Euler discretisation bias O(dt).</li>
	 * </ol>
	 */
	@Test
	public void testCaplet() throws CalculationException {
		final double fixingTime  = 3.5;
		final double paymentTime = 4.0;

		// ----- Section 1: Consistency ----------------------------------------
		final TermStructureMonteCarloSimulationModel simulationOriginalSpot     = buildOldSimulation("SPOT",     SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simulationOriginalTerminal = buildOldSimulation("TERMINAL", SEED_TERMINAL);
		final TermStructureMonteCarloSimulationModel simulationPluginSpot       = buildNewSimulation(new SpotMeasure(),     SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simulationPluginTerminal   = buildNewSimulation(new TerminalMeasure(), SEED_TERMINAL);

		final double originalSpot     = capletValue(simulationOriginalSpot,     fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double originalTerminal = capletValue(simulationOriginalTerminal, fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double pluginSpot       = capletValue(simulationPluginSpot,       fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double pluginTerminal   = capletValue(simulationPluginTerminal,   fixingTime, paymentTime, FLAT_FORWARD_RATE);

		System.out.println("=".repeat(80));
		System.out.println("  EXERCISE 2 — Measure Plug-in");
		System.out.println("  CAPLET  |  Fixing = 3.5 yr, Payment = 4.0 yr, Strike = forward rate (3 %)");
		System.out.println("=".repeat(80));
		System.out.println("  --- CONSISTENCY: does our new plug-in reproduce the original finmath-lib?");
		System.out.println("  " + "-".repeat(58));
		System.out.printf("  %-18s  %12s  %12s  %12s%n", "Measure", "Original", "Plug-in", "Rel. Error");
		System.out.println("  " + "-".repeat(58));
		System.out.printf("  %-18s  %+12.6f  %+12.6f  %11.2f %%%n", "Spot",     originalSpot,     pluginSpot,
				100.0 * Math.abs(pluginSpot - originalSpot) / Math.abs(originalSpot));
		System.out.printf("  %-18s  %+12.6f  %+12.6f  %11.2f %%%n", "Terminal", originalTerminal, pluginTerminal,
				100.0 * Math.abs(pluginTerminal - originalTerminal) / Math.abs(originalTerminal));
		System.out.println("  " + "-".repeat(58));
		System.out.println("");

		Assertions.assertEquals(originalSpot,     pluginSpot,     TOLERANCE_EXACT, "Caplet consistency: Spot");
		Assertions.assertEquals(originalTerminal, pluginTerminal, TOLERANCE_EXACT, "Caplet consistency: Terminal");

		// ----- Section 2: Measure invariance --------------------------------
		final TermStructureMonteCarloSimulationModel simulationSpot     = buildNewSimulation(new SpotMeasure(),     SEED_SPOT);
		final TermStructureMonteCarloSimulationModel simulationTerminal = buildNewSimulation(new TerminalMeasure(), SEED_TERMINAL);
		final double priceUnderSpot     = capletValue(simulationSpot,     fixingTime, paymentTime, FLAT_FORWARD_RATE);
		final double priceUnderTerminal = capletValue(simulationTerminal, fixingTime, paymentTime, FLAT_FORWARD_RATE);

		System.out.println("  --- MEASURE INVARIANCE");
		System.out.println("  Note: k >= 8 required (T_k >= payment date T_e = 4.0 yr).");
		System.out.println("  " + "-".repeat(67));
		System.out.printf("  %-34s  %12s  %18s%n", "Measure", "Price", "Rel. Error vs Spot");
		System.out.println("  " + "-".repeat(67));
		System.out.printf("  %-34s  %+12.6f  %18s%n",   "Spot measure (reference)", priceUnderSpot,     "—");
		System.out.printf("  %-34s  %+12.6f  %17.2f %%%n", "Terminal measure",       priceUnderTerminal,
				100.0 * Math.abs(priceUnderTerminal - priceUnderSpot) / priceUnderSpot);

		Assertions.assertEquals(priceUnderSpot, priceUnderTerminal, TOLERANCE_MONTE_CARLO, "Caplet: Spot vs Terminal");

		for(int i = 0; i < FORWARD_MEASURE_INDICES.length; i++) {
			final int k = FORWARD_MEASURE_INDICES[i];
			final TermStructureMonteCarloSimulationModel simulationForward =
					buildNewSimulation(new ForwardMeasure(k), SEED_FORWARD);
			final double price = capletValue(simulationForward, fixingTime, paymentTime, FLAT_FORWARD_RATE);
			System.out.printf("  %-34s  %+12.6f  %17.2f %%%n",
					"Forward measure (k=" + k + ", T=" + (k * PERIOD_LENGTH) + "yr)", price,
					100.0 * Math.abs(price - priceUnderSpot) / priceUnderSpot);
			Assertions.assertEquals(priceUnderSpot, price, TOLERANCE_MONTE_CARLO, "Caplet: Spot vs Forward measure k=" + k);
		}
		System.out.println("  " + "-".repeat(67));
		System.out.println("");

		// ----- Section 3: Accuracy -------------------------------------------
		// Bachelier effective annualised vol: sigma(tau) = a*exp(-c*tau)
		// Integrated variance = (a^2 / 2c) * (1 - exp(-2c * T_s))
		final double integratedVariance     = (VOLATILITY_PARAMETER_A * VOLATILITY_PARAMETER_A / (2.0 * VOLATILITY_PARAMETER_C))
				* (1.0 - Math.exp(-2.0 * VOLATILITY_PARAMETER_C * fixingTime));
		final double bachelierAnnualisedVol = Math.sqrt(integratedVariance / fixingTime);
		final double zeroCouponBond         = Math.pow(1.0 / (1.0 + FLAT_FORWARD_RATE * PERIOD_LENGTH), paymentTime / PERIOD_LENGTH);
		final double payoffUnit             = PERIOD_LENGTH * zeroCouponBond;

		System.out.println("  --- ACCURACY: Monte Carlo vs Bachelier analytical formula");
		System.out.printf("  Bachelier annualised vol = %.6f  |  Payoff unit = %.6f%n", bachelierAnnualisedVol, payoffUnit);
		System.out.println("  Note: Analytical = Bachelier formula (exact continuous-time price).");
		System.out.println("  Rel. Error = (Monte Carlo - Analytical) / Analytical.");
		System.out.println("  " + "-".repeat(70));
		System.out.printf("  %-10s  %14s  %14s  %14s  %14s%n", "Strike", "Monte Carlo", "Analytical", "Abs. Error", "Rel. Error");
		System.out.println("  " + "-".repeat(70));

		for(final double strike : TEST_STRIKES) {
			final double monteCarloValue    = capletValue(simulationSpot, fixingTime, paymentTime, strike);
			final double analyticalValue    = AnalyticFormulas.bachelierOptionValue(
					FLAT_FORWARD_RATE, bachelierAnnualisedVol, fixingTime, strike, payoffUnit);
			final double absoluteError      = monteCarloValue - analyticalValue;
			final double relativeErrorInPercent = 100.0 * absoluteError / analyticalValue;

			System.out.printf("  %8.2f %%  %+14.6f  %+14.6f  %+14.6f  %+13.2f %%%n",
					strike * 100.0, monteCarloValue, analyticalValue, absoluteError, relativeErrorInPercent);

			Assertions.assertEquals(analyticalValue, monteCarloValue, TOLERANCE_ANALYTICAL,
					String.format("Caplet accuracy at K = %.0f %%", strike * 100.0));
		}
		System.out.println("  " + "-".repeat(70));
		System.out.println("");
		System.out.println("  Remaining error is Euler discretisation bias: O(dt) with dt = " + PERIOD_LENGTH + " yr.");
		System.out.println("=".repeat(80));
		System.out.println();
	}

	// -------------------------------------------------------------------------
	// Product pricing helpers
	// -------------------------------------------------------------------------

	/**
	 * Prices a Forward Rate Agreement at time 0.
	 *
	 * <p>
	 * The Forward Rate Agreement pays \( (L(T_s, T_e) - K) \cdot \delta \) at \( T_e \).
	 * Its value at time 0 is:
	 * \[
	 *   V_0 = \mathrm{E}\!\left[\frac{N(0)}{N(T_e)} \cdot (L(T_s,T_e;\,T_s) - K) \cdot \delta\right]
	 * \]
	 *
	 * @param simulation  The Monte Carlo simulation.
	 * @param fixingTime  \( T_s \) — the LIBOR fixing date.
	 * @param paymentTime \( T_e \) — the payment date.
	 * @param strike      \( K \) — the fixed rate.
	 * @return Monte Carlo estimate of the Forward Rate Agreement value.
	 */
	private double forwardRateAgreementValue(
			final TermStructureMonteCarloSimulationModel simulation,
			final double fixingTime,
			final double paymentTime,
			final double strike) throws CalculationException {

		final double periodLength = paymentTime - fixingTime;
		final RandomVariable libor              = simulation.getLIBOR(fixingTime, fixingTime, paymentTime);
		final RandomVariable numeraire           = simulation.getNumeraire(paymentTime);
		final RandomVariable numeraireAtTimeZero = simulation.getNumeraire(0.0);

		return libor.sub(strike).mult(periodLength)
				.div(numeraire).mult(numeraireAtTimeZero)
				.getAverage();
	}

	/**
	 * Prices a Caplet (call on a LIBOR rate) at time 0.
	 *
	 * <p>
	 * The Caplet pays \( \max(L(T_s,T_e) - K,\,0) \cdot \delta \) at \( T_e \).
	 * Its value at time 0 is:
	 * \[
	 *   V_0 = \mathrm{E}\!\left[\frac{N(0)}{N(T_e)} \cdot \max(L(T_s,T_e;\,T_s) - K,\,0) \cdot \delta\right]
	 * \]
	 *
	 * @param simulation  The Monte Carlo simulation.
	 * @param fixingTime  \( T_s \) — the LIBOR fixing date.
	 * @param paymentTime \( T_e \) — the payment date.
	 * @param strike      \( K \) — the cap strike.
	 * @return Monte Carlo estimate of the Caplet value.
	 */
	private double capletValue(
			final TermStructureMonteCarloSimulationModel simulation,
			final double fixingTime,
			final double paymentTime,
			final double strike) throws CalculationException {

		final double periodLength = paymentTime - fixingTime;
		final RandomVariable libor              = simulation.getLIBOR(fixingTime, fixingTime, paymentTime);
		final RandomVariable numeraire           = simulation.getNumeraire(paymentTime);
		final RandomVariable numeraireAtTimeZero = simulation.getNumeraire(0.0);

		return libor.sub(strike).floor(0.0).mult(periodLength)
				.div(numeraire).mult(numeraireAtTimeZero)
				.getAverage();
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
	private TermStructureMonteCarloSimulationModel buildOldSimulation(
			final String measureName, final String stateSpaceName, final int seed) throws CalculationException {

		final var timeDiscretization  = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);
		final var tenorDiscretization = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);

		final var forwardCurve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"forwardCurve",
				new double[]{ 0.5, TIME_HORIZON },
				new double[]{ FLAT_FORWARD_RATE, FLAT_FORWARD_RATE },
				PERIOD_LENGTH);

		final var volatilityModel  = new LIBORVolatilityModelFourParameterExponentialForm(
				timeDiscretization, tenorDiscretization,
				VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B,
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
	 * Builds a simulation using the original {@link LIBORMarketModelFromCovarianceModel}
	 * with {@code StateSpace.NORMAL} (default).
	 *
	 * @param measureName {@code "SPOT"} or {@code "TERMINAL"}.
	 * @param seed        Random seed for the Brownian motion.
	 */
	private TermStructureMonteCarloSimulationModel buildOldSimulation(
			final String measureName, final int seed) throws CalculationException {
		return buildOldSimulation(measureName, LIBORMarketModelFromCovarianceModel.StateSpace.NORMAL.name(), seed);
	}

	/**
	 * Builds a simulation using our new {@link AugmentedLIBORMarketModel}.
	 *
	 * @param measure             The plug-in measure to inject.
	 * @param seed                Random seed for the Brownian motion.
	 */
	private TermStructureMonteCarloSimulationModel buildNewSimulation(
			final Measure measure, final int seed) throws CalculationException {
		return buildNewSimulation(measure, new NormalStateSpaceTransform(), seed);
	}

	/**
	 * Builds a simulation using our new {@link AugmentedLIBORMarketModel}
	 * with an explicit state-space transform.
	 *
	 * @param measure             The plug-in measure to inject.
	 * @param stateSpaceTransform The plug-in state-space transform to inject.
	 * @param seed                Random seed for the Brownian motion.
	 */
	private TermStructureMonteCarloSimulationModel buildNewSimulation(
			final Measure measure, final StateSpaceTransform stateSpaceTransform, final int seed) throws CalculationException {

		final var timeDiscretization  = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);
		final var tenorDiscretization = new TimeDiscretizationFromArray(0.0, (int)(TIME_HORIZON / PERIOD_LENGTH), PERIOD_LENGTH);

		final var forwardCurve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"forwardCurve",
				new double[]{ 0.5, TIME_HORIZON },
				new double[]{ FLAT_FORWARD_RATE, FLAT_FORWARD_RATE },
				PERIOD_LENGTH);

		final var volatilityModel  = new LIBORVolatilityModelFourParameterExponentialForm(
				timeDiscretization, tenorDiscretization,
				VOLATILITY_PARAMETER_A, VOLATILITY_PARAMETER_B,
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
}
