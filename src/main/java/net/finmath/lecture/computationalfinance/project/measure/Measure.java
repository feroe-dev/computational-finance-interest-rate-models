package net.finmath.lecture.computationalfinance.project.measure;

import net.finmath.exception.CalculationException;
import net.finmath.montecarlo.interestrate.LIBORMarketModel;
import net.finmath.montecarlo.process.MonteCarloProcess;
import net.finmath.stochastic.RandomVariable;
import net.finmath.lecture.computationalfinance.project.statespace.StateSpaceTransform;

/**
 * Plug-in interface for the probability measure used in a LIBOR Market Model simulation.
 *
 * <p>
 * In the LIBOR Market Model, the choice of probability measure affects two quantities:
 * <ul>
 *   <li>the <b>drift</b> \( \mu_j(t) \) of each forward rate \( L_j \), and</li>
 *   <li>the <b>numeraire</b> \( N(T_j) \) used for discounting.</li>
 * </ul>
 *
 * <p>
 * The drift of the \( j \)-th forward rate under a measure \( \mathbb{Q}^N \) is
 * \[
 *   \mu_j(t) = \sum_l \frac{\pm\,\delta_l}{1 + \delta_l L_l(t)} \,(\lambda_j(t) \cdot \lambda_l(t))
 * \]
 * where the summation range and sign depend on the chosen measure:
 * <ul>
 *   <li><b>Spot measure:</b> sum \( l = m(t){+}1,\ldots,j \) with a positive sign.</li>
 *   <li><b>Terminal measure \( T_n \):</b> sum \( l = j{+}1,\ldots,n{-}1 \) with a negative sign.</li>
 *   <li><b>\( T_k \)-forward measure:</b> sum \( l = j{+}1,\ldots,k{-}1 \) (negative) for \( j &lt; k \),
 *       or \( l = k,\ldots,j \) (positive) for \( j \geq k \).</li>
 * </ul>
 *
 * <p>
 * Implementations of this interface allow the model
 * {@code LIBORMarketModelFromCovarianceModelAndMeasure} to support arbitrary measures
 * without any modification to the model class itself (open/closed principle).
 *
 * @author Felipe, GM-1, GM-2
 */
public interface Measure {

	/**
	 * Returns the drift vector \( \mu(t_i) \) for all forward rate components at the
	 * given simulation time index, under this measure.
	 *
	 * <p>
	 * Components with index below {@code firstForwardRateIndex} are already fixed and
	 * must be left as {@code null} in the returned array. All other components must be
	 * non-null.
	 *
	 * <p>
	 * Note: the Ito correction \( -\frac{1}{2}\sigma_j^2 \) for log-normal state space is
	 * <em>not</em> part of this method — it is added by the model class after this call.
	 *
	 * @param process                The Monte-Carlo process (provides time grid and callbacks).
	 * @param timeIndex              The current simulation time index \( i \).
	 * @param firstForwardRateIndex  Index of the first forward rate that has not yet fixed.
	 * @param realizationAtTimeIndex Current forward rate vector \( L(t_i) \).
	 * @param stateSpaceTransform    The state-space transform plug-in (provides drift weight factor).
	 * @param model                  The LIBOR market model (provides period lengths and factor loadings).
	 * @return Drift vector \( \mu(t_i) \) — array of length {@code model.getNumberOfComponents()},
	 *         with {@code null} entries for already-fixed components.
	 */
	RandomVariable[] getDrift(
			MonteCarloProcess process,
			int timeIndex,
			int firstForwardRateIndex,
			RandomVariable[] realizationAtTimeIndex,
			StateSpaceTransform stateSpaceTransform,
			LIBORMarketModel model);

	/**
	 * Returns the numeraire \( N(T_j) \) at the LIBOR tenor time \( T_j \) given by
	 * {@code liborTimeIndex}.
	 *
	 * <p>
	 * The numeraire defines which pricing measure is used:
	 * <ul>
	 *   <li><b>Spot measure:</b>
	 *       \( N(T_j) = \prod_{l=0}^{j-1}(1 + \delta_l\,L_l(T_l)) \)</li>
	 *   <li><b>Terminal measure \( T_n \):</b>
	 *       \( N(T_j) = P(T_j,T_n) = \prod_{l=j}^{n-1} \frac{1}{1+\delta_l L_l(T_j)} \)</li>
	 *   <li><b>\( T_k \)-forward measure:</b>
	 *       \( N(T_j) = P(T_j,T_k) = \prod_{l=j}^{k-1} \frac{1}{1+\delta_l L_l(T_j)} \)</li>
	 * </ul>
	 *
	 * @param process        The Monte-Carlo process.
	 * @param liborTimeIndex The tenor index \( j \) in the LIBOR period discretization.
	 * @param model          The LIBOR market model (provides forward rates and period lengths).
	 * @return The numeraire \( N(T_j) \) as a {@link RandomVariable}.
	 * @throws CalculationException If the forward rates required for the computation are unavailable.
	 */
	RandomVariable getNumeraire(
			MonteCarloProcess process,
			int liborTimeIndex,
			LIBORMarketModel model) throws CalculationException;
}