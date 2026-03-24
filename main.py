import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)1.1s %(module)s:%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Hello from rgi-utils!")


if __name__ == "__main__":
    main()
