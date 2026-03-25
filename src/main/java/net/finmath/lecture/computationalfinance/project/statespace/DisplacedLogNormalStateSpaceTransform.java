package net.finmath.lecture.computationalfinance.project.statespace;

import net.finmath.stochastic.RandomVariable;

/**
 * Displaced log-normal state-space transform for the LIBOR Market Model.
 *
 * <p>
 * Each forward rate \( L_j \) is shifted by a per-index displacement \( a_j &gt; 0 \)
 * so that the <em>displaced</em> rate \( \tilde{L}_j = L_j + a_j \) is log-normally
 * distributed. The internal simulation state is:
 * \[
 *   X_j = \log(L_j + a_j), \qquad L_j = e^{X_j} - a_j.
 * \]
 *
 * <p>
 * This model interpolates between the two extremes:
 * <ul>
 *   <li>\( a_j \to 0 \): recovers the standard log-normal (Black) model.</li>
 *   <li>\( a_j \gg L_j \): approximates the normal (Bachelier) model.</li>
 * </ul>
 *
 * <p>
 * Rates can become negative, down to \( -a_j \), which is a key practical advantage
 * when modelling low or negative interest rate environments.
 *
 * <p>
 * The drift weight factor is \( g(L_j) = L_j + a_j \) (the displaced rate), and
 * the Itô correction is \( -\tfrac{1}{2}\sigma_j^2 \), identical to the standard
 * log-normal case applied to \( \log\tilde{L}_j \).
 *
 * @author Felipe, GM-1, GM-2
 * @see StateSpaceTransform
 * @see LogNormalStateSpaceTransform
 * @see NormalStateSpaceTransform
 */
public class DisplacedLogNormalStateSpaceTransform implements StateSpaceTransform {

    /** Per-index displacement parameters \( a_j \). */
    private final double[] displacement;

    /**
     * Creates a displaced log-normal transform with per-index displacements.
     *
     * @param displacement Array of displacement values \( a_j \), one per LIBOR index.
     */
    public DisplacedLogNormalStateSpaceTransform(final double[] displacement) {
        this.displacement = displacement;
    }

    /**
     * Creates a displaced log-normal transform with a uniform displacement across all indices.
     *
     * @param displacement   The common displacement value \( a \) applied to all LIBOR indices.
     * @param numberOfLibors The total number of LIBOR rates in the model.
     */
    public DisplacedLogNormalStateSpaceTransform(final double displacement, final int numberOfLibors) {
        this.displacement = new double[numberOfLibors];
        for (int i = 0; i < numberOfLibors; i++) {
            this.displacement[i] = displacement;
        }
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns \( X_j(0) = \log\!\max(L_j(0) + a_j,\, 10^{-10}) \).
     * The floor at \( 10^{-10} \) prevents \( \log(0) \) for zero or negative displaced rates.
     */
    @Override
    public double getInitialState(final double rate, final int liborIndex) {
        return Math.log(Math.max(rate + displacement[liborIndex], 1e-10)); // Avoid log(0)
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns \( L_j = e^{X_j} - a_j \).
     */
    @Override
    public RandomVariable applyTransform(final RandomVariable internalState, final int liborIndex) {
        return internalState.exp().sub(displacement[liborIndex]);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns \( X_j = \log(L_j + a_j) \).
     */
    @Override
    public RandomVariable applyInverseTransform(final RandomVariable rate, final int liborIndex) {
        return rate.add(displacement[liborIndex]).log();
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns the displaced rate \( L_j + a_j \), which is the extra factor
     * \( g(L_j) \) appearing in the drift sum under a displaced log-normal model.
     */
    @Override
    public RandomVariable getDriftWeightFactor(final RandomVariable liborRate, final int liborIndex) {
        return liborRate.add(displacement[liborIndex]);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns \( -\tfrac{1}{2}\,\sigma_j^2 \), identical to the standard log-normal
     * Itô correction, since the displaced rate \( \tilde{L}_j = L_j + a_j \) follows
     * a geometric Brownian motion.
     */
    @Override
    public RandomVariable getItoCorrection(final RandomVariable variance, final RandomVariable liborRate, final int liborIndex) {
        return variance.mult(-0.5);
    }

}
