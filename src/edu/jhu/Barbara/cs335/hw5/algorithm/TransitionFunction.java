package edu.jhu.Barbara.cs335.hw5.algorithm;

import java.io.Serializable;
import java.util.Set;

import edu.jhu.Barbara.cs335.hw5.data.Action;
import edu.jhu.Barbara.cs335.hw5.data.State;
import edu.jhu.Barbara.cs335.hw5.util.Pair;

/**
 * Implementers of this interface provide a function for state transition in the reinforcement learning world.
 * @author Zachary Palmer
 */
public interface TransitionFunction extends Serializable
{
	/**
	 * Provides legal transitions.
	 * @param state The state in question.
	 * @param action The action taken from that state.
	 * @return The possible outcomes of this transition.  These are represented as a pairing between the outcome state
	 *         and the probability of that outcome occurring.
	 */
	public Set<Pair<State,Double>> transition(State state, Action action);
}
