package net.finmath.lecture.computationalfinance.project.measure;

import net.finmath.exception.CalculationException;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;

/**
 * Real-world "riding the yield curve" measure for the LIBOR Market Model.
 *
 * <p>
 * Under this measure the drift of forward rate \( L_j \) is chosen so that
 * the entire initial term structure propagates forward at the speed of one
 * period per period:
 * \[
 *   \mu_j = \frac{f^{-1}(L_{j-1}(0)) - f^{-1}(L_j(0))}{\delta_j},
 * \]
 * where \( f^{-1} \) is the inverse state-space transform
 * (i.e. the initial internal state corresponding to the initial market rate).
 *
 * <p>
 * Consequently, after \( k \) time steps of length \( \delta \):
 * \[
 *   \mathbb{E}[L_j(T_k)] \approx L_{j-k}(0),
 * \]
 * meaning the yield curve is "ridden" forward in time — rate \( L_j \)
 * converges toward the initial short-end level as time passes.
 *
 * <p>
 * The numeraire is set to 1, consistent with simulation under the real-world
 * measure \( \mathbb{P} \).
 *
 * @author Felipe, GM-1, GM-2
 * @see Measure
 * @see ZeroDriftMeasure
 */
public class RidingTheCurveMeasure implements Measure {

	private final ForwardCurve forwardCurve;

	/**
	 * Creates a RidingTheCurveMeasure from the initial forward curve.
	 *
	 * @param forwardCurve The initial forward curve \( L_j(0) \).
	 */
	public RidingTheCurveMeasure(final ForwardCurve forwardCurve) {
		this.forwardCurve = forwardCurve;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>
	 * For component \( j \geq \texttt{firstForwardRateIndex} \) the drift is
	 * \[
	 *   \mu_j = \frac{X_{j-1}(0) - X_j(0)}{\delta_j}, \qquad X_j(0) = f^{-1}(L_j(0)),
	 * \]
	 * where the initial internal state \( X_j(0) \) is obtained from the
	 * state-space transform plug-in. For \( j = 0 \) the drift is zero.
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final int firstForwardRateIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final StateSpaceTransform stateSpaceTransform,
			final LIBORMarketModel model) {

		final int n = model.getNumberOfComponents();
		final RandomVariable[] drift = new RandomVariable[n];

		for(int j = firstForwardRateIndex; j < n; j++) {
			final double periodLength = model.getLiborPeriodDiscretization().getTimeStep(j);
			final double rateJ = forwardCurve.getForward(null, model.getLiborPeriod(j), periodLength);
			final double stateJ = stateSpaceTransform.getInitialState(rateJ, j);

			final double stateJm1;
			if(j > 0) {
				final double rateJm1 = forwardCurve.getForward(null, model.getLiborPeriod(j - 1), periodLength);
				stateJm1 = stateSpaceTransform.getInitialState(rateJm1, j - 1);
			} else {
				stateJm1 = stateJ; // j = 0: no predecessor, zero drift
			}

			drift[j] = model.getRandomVariableForConstant((stateJm1 - stateJ) / periodLength);
		}
		return drift;
	}

	/**
	 * {@inheritDoc}
	 *
	 * <p>Returns 1.0 so that pricing products compute raw expected cashflows
	 * without discounting.
	 */
	@Override
	public RandomVariable getNumeraire(
			final MonteCarloProcess process,
			final int liborTimeIndex,
			final LIBORMarketModel model) throws CalculationException {
		return model.getRandomVariableForConstant(1.0);
	}
}
