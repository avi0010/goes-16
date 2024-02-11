import argparse
import logging

from goes_16_date import GoesDownloaderDate, GoesDownloaderIndividualBboxDate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Type of Download")
    parser.add_argument("-s", "--save", required=True)
    parser.add_argument("-b", "--band", required=True)

    args = parser.parse_args()

    try:
        logging.info(f"Starting pre-processing of downloaded images")

        down = GoesDownloaderIndividualBboxDate(args.save)
        down.pre_processing(args.band)
        
        logging.info("Finished pre-processing")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise