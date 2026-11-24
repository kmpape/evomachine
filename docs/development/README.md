# EvoMachine developer guide

[Documentation](../README.md) › Developer guide

Use this guide when changing application behavior or adding a backend.

- [Architecture and code structure](architecture.md)
- [Configuration conventions and reference](configuration.md)
- [Peripherals, controllers, and bindings](bindings.md)
- [Commands, runtime lifecycle, and errors](runtime.md)
- [Virtual implementations](virtual-mode.md)
- [Extending the GUI](gui.md)
- [Extending AutoStrat](autostrat.md)
- [Image-processing integration](image-processing.md)
- [Development, testing, and releases](testing.md)

Start with architecture and configuration. A feature that crosses the GUI,
Automaton, and a binding should preserve the ownership boundaries described
there rather than bypassing a manager from the GUI.

---

← [Documentation home](../README.md) | [Architecture](architecture.md) →
