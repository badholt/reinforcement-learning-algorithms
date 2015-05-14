package edu.jhu.Barbara.cs335.hw5.algorithm;

import java.io.Serializable;

import edu.jhu.Barbara.cs335.hw5.data.Action;
import edu.jhu.Barbara.cs335.hw5.data.State;

/**
 * This interface is implemented by any class which represents a policy for a reinforcement agent.
 * @author Zachary Palmer
 */
public interface Policy extends Serializable
{
	/**
	 * Decides which action to take from a given state.
	 * @param state The current state of the agent.
	 */
	public Action decide(State state);
}
