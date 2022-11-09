import csv
import collections
import copy
import argparse
import os


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate statistic summary in format of HTML web page."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="directory where storing the csv files, exported by deepinsight benchmark runs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="output folder to save the HTML page.",
    )

    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args)

    directory_name = os.path.basename(os.path.dirname(args.input_dir))

    csv_file_paths = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ]
    all_item = {}
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_item[row["index"]] = row

    raw_dict = collections.OrderedDict(sorted(all_item.items()))

    def get_parent_idex(index):
        l = index.split(".")
        if len(l) <= 1:
            return None
            # raise ValueError("cannot get parent index for " + index)
        return ".".join(l[:-1])

    visited_keys = []
    updated_dict = {}

    root_key = list(raw_dict.keys())[0]
    for idx, k in enumerate(raw_dict.keys()):
        visited_keys.append(k)
        updated_dict[k] = raw_dict[k]
        updated_dict[k]["torch.fw_percent"] = (
            "{0:.2%}".format(
                float(updated_dict[k]["torch.fw_p0.8"])
                / float(raw_dict[root_key]["torch.fw_p0.8"])
            )
            if float(updated_dict[k]["torch.fw_p0.8"]) > 0
            else float(updated_dict[k]["torch.fw_p0.8"])
        )
        updated_dict[k]["torch.bw_percent"] = (
            "{0:.2%}".format(
                float(updated_dict[k]["torch.bw_p0.8"])
                / float(raw_dict[root_key]["torch.bw_p0.8"])
            )
            if float(updated_dict[k]["torch.bw_p0.8"]) > 0
            else float(updated_dict[k]["torch.bw_p0.8"])
        )
        updated_dict[k]["ortmodule.fw_percent"] = (
            "{0:.2%}".format(
                float(updated_dict[k]["ortmodule.fw_p0.8"])
                / float(raw_dict[root_key]["ortmodule.fw_p0.8"])
            )
            if float(updated_dict[k]["ortmodule.fw_p0.8"]) > 0
            else float(updated_dict[k]["ortmodule.fw_p0.8"])
        )
        updated_dict[k]["ortmodule.bw_percent"] = (
            "{0:.2%}".format(
                float(updated_dict[k]["ortmodule.bw_p0.8"])
                / float(raw_dict[root_key]["ortmodule.bw_p0.8"])
            )
            if float(updated_dict[k]["ortmodule.bw_p0.8"]) > 0
            else float(updated_dict[k]["ortmodule.bw_p0.8"])
        )
        if idx == 0:  # handle the root
            continue

        parent_index = get_parent_idex(k)
        while parent_index is not None and parent_index not in visited_keys:
            new_row = copy.deepcopy(raw_dict[k])
            visited_keys.append(parent_index)
            for row_key in new_row:
                new_row[row_key] = "-2.0"

            new_row["index"] = parent_index
            updated_dict[parent_index] = new_row
            parent_index = get_parent_idex(parent_index)

    raw_dict = collections.OrderedDict(sorted(updated_dict.items()))

    def get_depth(index):
        return len(index.split(".")) - 1

    def generate_record(raw_dict, d):
        current_str = (
            'index: "{}", signature: "{}", config:"{}", fw_pct:"{}", fw_mean: "{:.2f}", fw_0_8:"{:.2f}", bw_pct:"{}", bw_mean: "{:.2f}", bw_0_8: "{:.2f}",'
            + 'ort_fw_pct:"{}", ort_fw_mean:"{:.2f}", ort_fw_0_8:"{:.2f}", ort_bw_pct:"{}", ort_bw_mean: "{:.2f}", ort_bw_0_8: "{:.2f}",'
            + 'diff_fw_mean:"{}", diff_fw_0_8:"{}", diff_bw_mean: "{}", diff_bw_0_8: "{}"'
        ).format(
            d["index"],
            d["fn(inputs)[signature]"],
            d["input & run config"],
            d["torch.fw_percent"],
            float(d["torch.fw_mean"]),
            float(d["torch.fw_p0.8"]),
            d["torch.bw_percent"],
            float(d["torch.bw_mean"]),
            float(d["torch.bw_p0.8"]),
            d["ortmodule.fw_percent"],
            float(d["ortmodule.fw_mean"]),
            float(d["ortmodule.fw_p0.8"]),
            d["ortmodule.bw_percent"],
            float(d["ortmodule.bw_mean"]),
            float(d["ortmodule.bw_p0.8"]),
            d["diff.fw_mean"],
            d["diff.fw_p0.8"],
            d["diff.bw_mean"],
            d["diff.bw_p0.8"],
        )

        children = []
        current_indend = get_depth(d["index"])
        for k in raw_dict.keys():
            # print(k, k.startswith(d['index'] + "."), get_depth(k), current_indend)
            if k.startswith(d["index"] + ".") and get_depth(k) == current_indend + 1:
                children.append(k)

        # print(children, d['index'])

        if len(children) > 0:
            current_str += ", _children: ["
            for child_name in children:
                current_str += generate_record(raw_dict, raw_dict[child_name])
            current_str += "]"

        return "{" + current_str + "},"

    def generate_data_from_root(raw_dict, root_id):
        return generate_record(raw_dict, raw_dict[root_id])

    data_str = generate_data_from_root(raw_dict, "root.0")

    """
    Simple - A plain, simplistic layout showing only basic grid lines. (/themes/tabulator_simple.css)
    Midnight - A dark, stylish layout using simple shades of grey. (/themes/tabulator_midnight.css)
    Modern - A neat, stylish layout using one primary color. (/themes/tabulator_modern.css)
    Site - The theme used for tables on this site.

    <link href="https://unpkg.com/tabulator-tables@5.0.10/dist/css/tabulator.min.css" rel="stylesheet">
    """

    html_template = """<html>
    <head>
    <title>"""
    html_template += "{}".format(directory_name)
    html_template += """</title>
        <link href="https://unpkg.com/tabulator-tables@5.0.10/dist/css/tabulator_site.min.css" rel="stylesheet">
        <link href="http://tabulator.info/css/app.css" rel="stylesheet">
        <script type="text/javascript" src="https://unpkg.com/tabulator-tables@5.0.10/dist/js/tabulator.min.js"></script>

        <script type="text/javascript" src="https://oss.sheetjs.com/sheetjs/xlsx.full.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.4.0/jspdf.umd.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.20/jspdf.plugin.autotable.min.js"></script>
    </head>
    <body style='padding: 20px; margin:auto'>
    """

    html_template += "<p>Aggregated results for metrics from {}</p>".format(
        csv_file_paths
    )

    html_template += """
        <div class="table-controls-legend">Download Controls</div>
        <div class='table-controls'>
            <button id="download-csv">Download CSV</button>
            <button id="download-json">Download JSON</button>
            <button id="download-xlsx">Download XLSX</button>
            <button id="download-pdf">Download PDF</button>
            <button id="download-html">Download HTML</button>
        </div>

        <div id="example-table"></div>
        <script>
            var tableDataNested = [
    """
    html_template += data_str

    html_template += """
            ];


            function diff_value_formatter(cell, formatterParams, onRendered){
                    //cell - the cell component
                    //formatterParams - parameters set for the column
                    //onRendered - function to call when the formatter has been rendered
                    val = parseFloat(cell.getValue()) / 100
                    if (val > 0.30) {
                        cell.getElement().style.backgroundColor = "#f19072";
                    } else if (val > 0.10) {
                        cell.getElement().style.backgroundColor = "#f7b977";
                    }  else if (val > 0.05) {
                        cell.getElement().style.backgroundColor = "#efcd9a";
                    }
                    return cell.getValue(); //return the contents of the cell;
            }

            var table = new Tabulator("#example-table", {
                height:"100%",
                data:tableDataNested,
                dataTree:true,
                dataTreeStartExpanded : true,
                //dataTreeChildColumnCalcs:true, //include child rows in column calculations
                //groupBy: "index",
                columnHeaderVertAlign:"bottom", //align header contents to bottom of cell
                columnDefaults:{
                    tooltip:true,
                },
                columns:
                [
                    {
                        title:"index", field:"index", responsive:0
                    }, //never hide this column
                    {
                        title:"signature", field:"signature", width:200
                    },
                    {
                        title:"config", field:"config", width:50, responsive:2
                    }, //hide this column first
                    {
                        title:"torch (us)",
                        columns:[
                            {
                                title:"forward",
                                columns:[
                                    {title:"pct", field:"fw_pct"},
                                    {title:"mean", field:"fw_mean"},
                                    {title:"0.8pctl", field:"fw_0_8"},
                                ]
                            },
                            {
                                title:"backward",
                                columns:[
                                    {title:"pct", field:"bw_pct"},
                                    {title:"mean", field:"bw_mean"},
                                    {title:"0.8pctl", field:"bw_0_8"},
                                ]
                            }
                        ]
                    },
                    {
                        title:"ortmodule (us)",
                        columns:[
                            {
                                title:"forward",
                                columns:[
                                    {title:"pct", field:"ort_fw_pct"},
                                    {title:"mean", field:"ort_fw_mean"},
                                    {title:"0.8pctl", field:"ort_fw_0_8"},
                                ]
                            },
                            {
                                title:"backward",
                                columns:[
                                    {title:"pct", field:"ort_bw_pct"},
                                    {title:"mean", field:"ort_bw_mean"},
                                    {title:"0.8pctl", field:"ort_bw_0_8"},
                                ]
                            }
                        ]
                    },
                    {
                        title:"diff",
                        columns:[
                            {
                                title:"forward",
                                columns:[
                                    {title:"mean", field:"diff_fw_mean", formatter: diff_value_formatter},
                                    {title:"0.8pctl", field:"diff_fw_0_8", formatter: diff_value_formatter},
                                ]
                            },
                            {
                                title:"backward",
                                columns:[
                                    {title:"mean", field:"diff_bw_mean", formatter: diff_value_formatter},
                                    {title:"0.8pctl", field:"diff_bw_0_8", formatter: diff_value_formatter},
                                ]
                            }
                        ]
                    }
                ],
            });


        //trigger download of data.csv file
        document.getElementById("download-csv").addEventListener("click", function(){
            table.download("csv", "data.csv");
        });

        //trigger download of data.json file
        document.getElementById("download-json").addEventListener("click", function(){
            table.download("json", "data.json");
        });

        //trigger download of data.xlsx file
        document.getElementById("download-xlsx").addEventListener("click", function(){
            table.download("xlsx", "data.xlsx", {sheetName:"My Data"});
        });

        //trigger download of data.pdf file
        document.getElementById("download-pdf").addEventListener("click", function(){
            table.download("pdf", "data.pdf", {
                orientation:"portrait", //set page orientation to portrait
                title:"Example Report", //add title to report
            });
        });

        //trigger download of data.html file
        document.getElementById("download-html").addEventListener("click", function(){
            table.download("html", "data.html", {style:true});
        });

        </script>
    </body>
    </html>
    """

    # to open/create a new html file in the write mode
    f = open(os.path.join(args.output_dir, directory_name + ".html"), "w")
    # writing the code into the file
    f.write(html_template)
    # close the file
    f.close()


if __name__ == "__main__":
    main()
