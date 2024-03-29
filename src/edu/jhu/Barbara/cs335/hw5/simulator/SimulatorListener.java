package edu.jhu.Barbara.cs335.hw5.simulator;

/**
 * This interface is implemented by any class which wishes to be notified of simulator events.
 * @author Zachary Palmer
 */
public interface SimulatorListener
{
	/**
	 * Indicates that a simulation event has occurred.
	 * @param event An event object describing the event that occurred.
	 */
	public void simulationEventOccurred(SimulatorEvent event);
}
