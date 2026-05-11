class ServiceRegistry:
    """Lightweight registry mapping service names to instances.

    Routes and services that need a cross-service reference use
    registry.get("name") rather than importing the module directly,
    which keeps the dependency graph explicit and cycle-free.
    """

    def __init__(self) -> None:
        self._services: dict = {}

    def register(self, name: str, service) -> None:
        self._services[name] = service

    def get(self, name: str):
        return self._services.get(name)


# Module-level singleton
registry = ServiceRegistry()
