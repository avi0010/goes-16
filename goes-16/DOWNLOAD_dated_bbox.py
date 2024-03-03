import argparse
import logging

from goes_16_date import GoesDownloaderDate, GoesDownloaderIndividualBboxDate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Type of Download")
    parser.add_argument("-s", "--save", required=True)
    parser.add_argument("-p", "--param", required=True)
    parser.add_argument("-d", "--prev-days", required=False)

    args = parser.parse_args()

    try:
        logging.info(f"Bulk Downloading based on bbox geojson start & end dates")

        down = GoesDownloaderIndividualBboxDate(args.save, args.prev_days)

        # Removing older downloaded data
        #down.clean_root_dir(args.param)

        # Download files b/w the extremeties of bboxes
        down.download(down.start, down.end, args.param)
        logging.info("Finished downloading")
    except Exception as e:
        logging.error(e, exc_info=True)
        #down.clean_root_dir(args.param)
        logging.warning("Error occured- Cleaning all downloaded images")
        raise