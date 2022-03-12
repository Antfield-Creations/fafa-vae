import argparse
import logging
import math
from os.path import isfile
from time import sleep
from typing import List

import requests
from tqdm import tqdm
from retry import retry
from bs4 import BeautifulSoup
from os import makedirs

from urllib.request import urlretrieve

base_href = 'https://www.female-anatomy-for-artist.com'
logging.basicConfig(level=logging.INFO)

# Without any browser info, the site hands out pages with 30 items per page
photos_per_page = 30
pause_every = 500

session = requests.session()


def main(args: argparse.Namespace) -> None:
    first_set = int(args.first_set)
    last_set = int(args.last_set)

    with tqdm(total=last_set) as pbar:
        # Skip over the part that is already done
        pbar.update(n=first_set)

        for set_no in range(first_set, last_set + 1):
            harvest_set(pbar, set_no, args.directory)

    session.close()


def harvest_set(pbar: tqdm, set_no: int, download_folder: str) -> None:
    """
    Harvests large-size thumbs for an entire FAFA photo set

    :param pbar:            A tqdm progress bar instance
    :param set_no:          The set number
    :param download_folder: Folder to store the thumbnail images

    :return: None
    """

    set_url = f'{base_href}/photos/showSet/id/{set_no}/thumb/small'
    set_folder = download_folder + f'/set-{set_no}'
    makedirs(set_folder, exist_ok=True)

    # First get the contents of the photo set page
    page = retry_get(set_url)

    # Validate OK response
    if not page.ok:
        logging.error(f'Skipped set {set_no}')
        return

    soup = BeautifulSoup(page.text, features='html.parser')
    search_hint_tag = soup.find(class_='searchHint')

    # Validate that the number of images for the set was advertised on the page
    if not hasattr(search_hint_tag, 'text') or search_hint_tag is None:
        logging.error(f'Set {set_no} had no list of images, skipping')
        return

    # The number of photos in the set is in the text before the first space, after the comma
    first_space_pos = search_hint_tag.text.find(' ')
    num_photos = search_hint_tag.text[1:first_space_pos]
    num_pages = math.ceil(int(num_photos) / photos_per_page)

    photo_page_links: List[str] = []

    for page_num in range(1, num_pages + 1):
        links = get_subpage_links(set_url, page_num)
        photo_page_links.extend(links)

    for idx, photo_page_href in enumerate(photo_page_links):
        harvest_image(photo_page_href, set_folder)
        pbar.set_description(f'Set {set_no}: photo {idx} of max {len(photo_page_links)}')

    # Advance progress bar
    pbar.update()


def harvest_image(photo_page_href, set_folder) -> None:
    """
    Harvests an image from a FAFA image page

    :param photo_page_href: Link to the photo page
    :param set_folder:      Folder for the set to store the image

    :return: None
    """
    page = retry_get(photo_page_href)

    if not page.ok:
        logging.error(f'Skipped photo {photo_page_href}: {page.status}')
        return

    soup = BeautifulSoup(page.text, features='html.parser')
    img_tag = soup.find(id='mainImage')
    if img_tag is None:
        logging.error('No image tag')
        return

    img_link = img_tag.attrs['src']
    photo_id = photo_page_href.split('/')[-1]
    file_name = photo_id + '-' + img_link.split('/')[-1]
    download_path = set_folder + '/' + file_name

    if isfile(download_path):
        logging.info(f'Already have {file_name}')
        return

    # Pause every now and then
    if int(photo_id) % pause_every == 0:
        logging.info('Pausing to prevent ip bans')
        session.close()
        sleep(180)

    retry_download(download_path, img_link)


def get_subpage_links(set_url: str, page_num: int) -> List[str]:
    page_url = f'{set_url}/page/{page_num}'
    page = retry_get(page_url)

    if not page.ok:
        logging.error(f'Skipped page {page_url}: bad response {page.status_code}')
        return []

    soup = BeautifulSoup(page.text, features='html.parser')
    screenshots = soup.find_all(class_='screenshot')
    photo_page_links = [base_href + ss.attrs['href'] for ss in screenshots]

    return photo_page_links


@retry(tries=5, delay=1, backoff=2)
def retry_get(page_url):
    page = session.get(page_url)

    if not page.ok:
        raise ValueError(f"Error getting {page_url}")

    return page


@retry(tries=3, delay=1, backoff=2)
def retry_download(download_path, img_link):
    urlretrieve(img_link, download_path)
    logging.debug(f'Wrote {download_path} from {img_link}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='FAFA image thumbnail harvester')
    argparser.add_argument('-d', '--directory', help='The destination dir to store the thumbs', required=True)
    argparser.add_argument('-f', '--first-set', help='The first set number to start harvesting at', default='0')
    argparser.add_argument('-l', '--last-set', help='The last set number to stop harvesting at', default='6000')
    args = argparser.parse_args()

    main(args)
