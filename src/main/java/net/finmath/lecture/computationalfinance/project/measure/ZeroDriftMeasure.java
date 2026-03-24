package net.finmath.lecture.computationalfinance.project.measure;

import net.finmath.exception.CalculationException;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;

/**
 * Real-world measure with zero drift for the LIBOR Market Model.
 *
 * <p>
 * Under this measure every forward rate \( L_j \) is a martingale:
 * \[
 *   \mathrm{d}L_j(t) = \sigma_j(t)\,\mathrm{d}W_j(t), \qquad \mu_j \equiv 0.
 * \]
 * Consequently \( \mathbb{E}[L_j(T)] = L_j(0) \) for all \( T \geq 0 \).
 *
 * <p>
 * The numeraire is set to 1 so that {@code product.getValue(simulation)} returns
 * the raw (undiscounted) expected cashflow under \( \mathbb{P} \).
 *
 * @author Felipe, GM-1, GM-2
 * @see Measure
 * @see RidingTheCurveMeasure
 */
public class ZeroDriftMeasure implements Measure {

	/**
	 * {@inheritDoc}
	 *
	 * <p>Returns zero drift for all live forward rate components.
	 */
	@Override
	public RandomVariable[] getDrift(
			final MonteCarloProcess process,
			final int timeIndex,
			final int firstForwardRateIndex,
			final RandomVariable[] realizationAtTimeIndex,
			final StateSpaceTransform stateSpaceTransform,
			final LIBORMarketModel model) {

		final RandomVariable zero = model.getRandomVariableForConstant(0.0);
		final RandomVariable[] drift = new RandomVariable[model.getNumberOfComponents()];
		for(int j = firstForwardRateIndex; j < drift.length; j++) {
			drift[j] = zero;
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
