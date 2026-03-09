"""
Author: zhengnianzu
Place: shenzhen
Time: 2020.8.25
Update: 2022.10.17
"""
from ast import arg
from distutils.command.config import config
from inspect import ArgSpec
import os
import sys
import base64
import argparse
import shutil


def add_description(title, description):
    return f"""<p><b>{title}</b>{description}</p>"""


def add_header(header_list):
    head_str = "<thead><tr>\n"
    for i, h in enumerate(header_list):
        if i == 0:
            head_str += f"""\t<th style=text_align: center;width:400> {h} </th> \n"""  # <td colspan="2"></td>
            continue
        head_str += f"""\t<th style=text_align: center;width:100>{h}</th>\n"""
    head_str += "</tr><thead>\n"
    return head_str


def add_row(text, wav_list):
    row_str = f"""<tr><td style=text_align: center;width:400>{text}</td>\n"""
    for wav_path in wav_list:
        row_str += f"""\t<td style=text_align: center;width:100>""" + \
                   f"""<audio src="{wav_path}" type="audio/mpeg" controls="" controlsList=nodownload>""" + \
                   f"""Your browser does not support the audio element.</audio></td> \n"""
    row_str += "</tr>\n"
    return row_str


def read_configs(config):
    content = []
    for line in open(config):
        ln = line.strip().split("|")
        content.append([v.strip() for v in ln])
    return content


def make_table(config_text=None, config_list=None, max_rows=20, data_dir=None):
    if data_dir is None:
        input_dir = os.path.dirname(config_text)
    else:
        input_dir = data_dir

    if config_list is None:
        content = read_configs(config_text)
    else:
        content = config_list

    data = os.path.join(input_dir, 'data')
    if not os.path.exists(data):
        os.makedirs(data, exist_ok=True)

    table_str = "<table>"
    table_str += add_header(content[0])
    table_str += "<tbody>"
    row_number = 1
    for row_line in content[1:]:
        nw = []
        for w in row_line[1:]:
            n = os.path.basename(w)
            nn = os.path.join(data, n)
            shutil.copyfile(w, nn)
            n_ = os.path.join('data', n)
            nw.append(n_)

        table_str += add_row(row_line[0], nw)

        row_number += 1
        if row_number > max_rows:
            break

    table_str += "</tbody>"
    table_str += "</table>"
    return table_str


def wav_to_html(file_path):
    try:
        with open(file_path, 'rb') as bf:
            bf_data = bf.read()
            base64_data = base64.b64encode(bf_data)  # get base64 string
            base64_message = base64_data.decode("utf-8")
        return "data:audio/wav;base64, {}".format(base64_message)
    except Exception as e:
        return ''


def add_embed_row(text, wav_list, wavesurfer=True, row_number=1):
    row_str = f"""<tr><td style=text_align: center;width:400>{text}</td>\n"""
    i = 1
    for wav_path in wav_list:
        wav_html = wav_to_html(wav_path)
        if wav_html is not None:
            if not wavesurfer:
                row_str += f"""\t<td style=text_align: center;width:100>""" + \
                           f"""<audio src="{wav_html}" type="audio/mpeg" controls="" controlsList=nodownload>""" + \
                           f"""Your browser does not support the audio element.</audio></td> \n"""
            else:
                indx = f"{row_number}_{i}"
                row_str += f"""\t<td style=text_align: center;width:100>""" + \
                           f"""{wavesurfer_cell(wav_path, indx)}""" + \
                           f"""</td> \n"""

        else:
            row_str += f"""\t<td style=text_align: center;width:100> </td> \n"""
        i += 1
    row_str += "</tr>\n"
    return row_str


def wavesurfer_cell(fpath, i=1):
    wstr = wav_to_html(fpath)
    wstr = f"""
    <div id="prompts_{i}_header_waveform"></div>
    <button id="prompts_{i}_header" class="play-button-demo btn btn-primary" onclick="wavesurfer_prompts_{i}.playPause()">
        <i class="fa fa-play"></i>
        Play
        /
        <i class="fa fa-pause"></i>
        Pause
    </button>
    <script>
        var wavesurfer_prompts_{i} = WaveSurfer.create(
            {{
                    container: '#prompts_{i}_header_waveform',
                    waveColor: 'violet',
                    progressColor: 'purple'
                }});
    """ + f"""
        wavesurfer_prompts_{i}.load("{wstr}");
    </script>
    """
    return wstr


def make_embed_table(config_text=None, config_list=None, wavesurfer=False, max_rows=100):
    if config_list is None:
        content = content = read_configs(config_text)
    else:
        content = config_list

    table_str = "<table>"
    table_str += add_header(content[0])
    table_str += "<tbody>"
    row_number = 1
    for row_line in content[1:]:
        table_str += add_embed_row(row_line[0], row_line[1:], wavesurfer=wavesurfer, row_number=row_number)
        row_number += 1
        if row_number > max_rows:
            break

    table_str += "</tbody>"
    table_str += "</table>"
    return table_str


def make_html(args):
    if not args.split:
        tbStr = make_embed_table(config_text=args.config, wavesurfer=args.wave_surf, max_rows=args.max_rows)
    else:
        tbStr = make_table(config_text=args.config, max_rows=args.max_rows, data_dir=os.path.dirname(args.name))

    template = """\
            <html>
                <head>
                  <title>Audio Samples</title>
                  <meta name="description" content="audio web show">
                  <meta charset="utf-8">
                  <style>
                    table {
                        page-break-inside: auto;
                    }
                    tr {
                        page-break-inside: avoid;
                        page-break-after: auto;
                    }
                    td {
                        word-wrap: break-word;
                        max-width: 150px;
                    }
                    thead {
                        display: table-header-group;
                    }
                    tfoot {
                        display: table-footer-group;
                    }
                  </style>
                  <script src="https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/1.2.3/wavesurfer.min.js"></script>
                </head>
                """
    template += """

                <body>
                    <h1>Audio Samples</h1>
                    <div id='section-1' class='section'>
                     <p>{}</p>
                    </div>
                </body>
            </html>
            """.format(tbStr)

    with open(args.name, "w") as g:
        g.write(template)


if __name__ == "__main__":
    """
    args:
    embd
    html_name
    data_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='eval.html')
    parser.add_argument('-s', '--split', action='store_true')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--wave-surf', action='store_true')
    parser.add_argument('--max_rows', type=int, default=100)
    args = parser.parse_args()
    make_html(args)
