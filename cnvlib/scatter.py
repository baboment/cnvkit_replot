"""The 'scatter' command for rendering copy number as scatter plots."""
import collections
import logging
from matplotlib.lines import Line2D
from matplotlib import colors, cm

import numpy as np
from matplotlib import pyplot
from skgenome.rangelabel import unpack_range

from . import core, params, plots
from .plots import MB
from .cnary import CopyNumArray as CNA

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rcParams

# Prefer Arial; fall back to Liberation Sans / DejaVu Sans if Arial isn't installed
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,   # nicer minus sign rendering
})


HIGHLIGHT_COLOR = "gold"
POINT_COLOR = "#AAA8A8"
SEG_COLOR = "darkorange"
TREND_COLOR = "#6A6A6A"
SEG_CMAP = pyplot.get_cmap("seismic")
SEG_COLORS = [
    (-1.1, "#053061"),
    (-0.5, "#2166ac"),
    (-0.25, "#4393c3"),
    (0.0, TREND_COLOR),
    (0.25, "#d6604d"),
    (0.5, "#b2182b"),
    (1, "#67001f"),
]

def _color_from_log2(value, norm=None):
    """Return a color based on the log2 value."""
    if norm is not None:
        return SEG_CMAP(norm(value))
    for cutoff, color in SEG_COLORS:
        if value >= cutoff:
            return color
    return SEG_COLORS[-1][1]


def do_scatter(
    cnarr,
    segments=None,
    variants=None,
    show_range=None,
    show_gene=None,
    do_trend=False,
    by_bin=False,
    window_width=1e6,
    y_min=None,
    y_max=None,
    fig_size=(40, 3),
    antitarget_marker=None,
    segment_color=SEG_COLOR,
    title=None,
):
    """Plot probe log2 coverages and segmentation calls together."""
    if by_bin:
        bp_per_bin = sum(c.end.iat[-1] for _, c in cnarr.by_chromosome()) / len(cnarr)
        window_width /= bp_per_bin
        show_range_bins = plots.translate_region_to_bins(show_range, cnarr)
        cnarr, segments, variants = plots.update_binwise_positions(
            cnarr, segments, variants
        )
        global MB
        orig_mb = MB
        MB = 1

    if not show_gene and not show_range:
        fig = genome_scatter(
            cnarr, segments, variants, do_trend, y_min, y_max, title, segment_color
        )
    else:
        if by_bin:
            show_range = show_range_bins
        fig = chromosome_scatter(
            cnarr,
            segments,
            variants,
            show_range,
            show_gene,
            antitarget_marker,
            do_trend,
            by_bin,
            window_width,
            y_min,
            y_max,
            title,
            segment_color,
        )

    if by_bin:
        # Reset to avoid permanently altering the value of cnvlib.scatter.MB
        MB = orig_mb
    if fig_size:
        width, height = fig_size
        fig.set_size_inches(w=width, h=height)
    return fig


# === Genome-level scatter plots ===


def genome_scatter(
    cnarr,
    segments=None,
    variants=None,
    do_trend=False,
    y_min=None,
    y_max=None,
    title=None,
    fig_size=(40, 4),
    segment_color=SEG_COLOR,
):
    """Plot all chromosomes, concatenated on one plot."""
    if (cnarr or segments) and variants:
        # Lay out top 3/5 for the CN scatter, bottom 2/5 for SNP plot
        axgrid = pyplot.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.85)
        axis = pyplot.subplot(axgrid[0])
        axis2 = pyplot.subplot(axgrid[1], sharex=axis)
        # Place chromosome labels between the CNR and SNP plots
        axis2.tick_params(labelbottom=False)
        chrom_sizes = plots.chromosome_sizes(cnarr or segments)
        axis2 = snv_on_genome(
            axis2, variants, chrom_sizes, segments, do_trend, segment_color
        )
    else:
        _fig, axis = pyplot.subplots()
    if title is None:
        title = (cnarr or segments or variants).sample_id
    if cnarr or segments:
        axis = cnv_on_genome(
            axis, cnarr, segments, do_trend, y_min, y_max, segment_color, title
        )
    else:
        axis.set_title(f"Variant allele frequencies: {title}", fontsize=14)
        chrom_sizes = collections.OrderedDict(
            (chrom, subarr["end"].max()) for chrom, subarr in variants.by_chromosome()
        )
        axis = snv_on_genome(
            axis, variants, chrom_sizes, segments, do_trend, segment_color
        )
    return axis.get_figure()


def cnv_on_genome(
    axis,
    probes,
    segments,
    do_trend=False,
    y_min=None,
    y_max=None,
    segment_color=SEG_COLOR,
    title=None,
    fig_size=(40, 4),
):
    """Plot bin ratios and segments for all chromosomes on one plot, with:
       - bottom chromosome labels (1..22,X,Y), bigger font
       - blue points for log2 >= 0, red for log2 < 0
       - dark-grey segment lines
       - no top/bottom spines (clean look)
    """
    # y axis label unchanged
    axis.set_ylabel("Copy ratio (log2)", fontsize=20)

    # ---- y-limits (auto if not provided)
    if not (y_min and y_max):
        if segments is not None and len(segments):
            low_chroms = segments.chromosome.isin(("6", "chr6", "Y", "chrY"))
            seg_auto_vals = segments[~low_chroms]["log2"].dropna()
            if not y_min:
                y_min = (
                    np.nanmin([seg_auto_vals.min() - 0.2, -1.5])
                    if len(seg_auto_vals)
                    else -2.5
                )
            if not y_max:
                y_max = (
                    np.nanmax([seg_auto_vals.max() + 0.2, 1.5])
                    if len(seg_auto_vals)
                    else 2.5
                )
        else:
            y_min = y_min or -2.5
            y_max = y_max or  2.5
    y_min, y_max = -2.0, 2.0
    axis.set_ylim(y_min, y_max)

    # ---- chromosome sizes & grouping
    if probes:
        chrom_sizes = plots.chromosome_sizes(probes)
        chrom_probes = dict(probes.by_chromosome())
    else:
        chrom_sizes = plots.chromosome_sizes(segments)
        chrom_probes = {}

    chrom_segs = dict(segments.by_chromosome()) if segments else {}

    # ---- create padded layout & dividers
    pad = 0.003 * sum(chrom_sizes.values())
    x_starts = collections.OrderedDict()
    curr_offset = pad
    label_positions = []  # (center_pos, chrom_name)
    for chrom, size in chrom_sizes.items():
        x_starts[chrom] = curr_offset
        axis.axvline(x=curr_offset, color="#000000", linewidth=1.2, zorder=20)
        if probes and chrom in chrom_probes:
            subprobes = chrom_probes[chrom]
            pos = 0.5 * (subprobes["start"].min() + subprobes["end"].max()) + curr_offset
        else:
            pos = curr_offset + 0.5 * size
        label_positions.append((pos, chrom))
        curr_offset += size + 2 * pad
    if label_positions:
        axis.axvline(x=curr_offset - pad, color="#000000", linewidth=1.2, zorder=20)
    axis.set_xlim(0, curr_offset)

    # ---- scatter: blue for >=0, red for <0
    for chrom, x_offset in x_starts.items():
        if probes and chrom in chrom_probes:
            subprobes = chrom_probes[chrom]
            x = 0.5 * (subprobes["start"] + subprobes["end"]) + x_offset
            y = subprobes["log2"].values
            # color by sign
            c = np.where(y >= 0.0, "#1A35FF", "#FF1A34")  # blue / red
            axis.scatter(
                x, y,
                c=c,
                marker=".",
                edgecolor="none",
                alpha=0.5,
                s=18,
                rasterized=True,
            )
            if do_trend:
                axis.plot(
                    x,
                    subprobes.smooth_log2(),
                    color=POINT_COLOR,
                    linewidth=2,
                    zorder=-1,
                    snap=False,
                )

    # ---- segments: all dark grey
    for chrom, x_offset in x_starts.items():
        for seg in chrom_segs.get(chrom, []):
            axis.plot(
                (seg.start + x_offset, seg.end + x_offset),
                (seg.log2, seg.log2),
                color="#585858",     # dark grey for every segment
                linewidth=5,
                solid_capstyle="round",
                snap=False,
                zorder=10,
            )

    # ---- bottom chromosome labels, larger; no Mb ticks
    tick_positions = [p for p, _ in label_positions]
    tick_labels = [
        (name[3:] if name.lower().startswith("chr") else name)
        for _, name in label_positions
    ]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, fontsize=20)  # bigger labels
    axis.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    axis.tick_params(axis="y", labelsize=16)
    # remove any minor/major Mb ticks if they were set earlier
    axis.set_xticks([], minor=True)

    # ---- clean frame: remove top & bottom black lines (spines)
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    # keep left/right so y-axis & dividers remain readable
    axis.spines["right"].set_visible(False)

    # no colorbar here (all segments are dark grey; points are fixed blue/red)
    pyplot.subplots_adjust(bottom=0.18, left=0.06, right=0.98, top=0.90)
    return axis

# === Chromosome-level scatter plots ===


def chromosome_scatter(
    cnarr,
    segments,
    variants,
    show_range,
    show_gene,
    antitarget_marker,
    do_trend,
    by_bin,
    window_width,
    y_min,
    y_max,
    title,
    segment_color,
):
    """Plot a specified region on one chromosome.

    Possibilities::

             Options | Shown
        ------------ | --------
        -c      | -g | Genes | Region
        ------- | -- | ----- | ------
        -       | +  | given | auto: gene(s) + margin
        chr     | -  | none  | whole chrom
        chr     | +  | given | whole chrom
        chr:s-e | -  | all   | given
        chr:s-e | +  | given | given

    """
    sel_probes, sel_segs, sel_snvs, window_coords, genes, chrom = select_range_genes(
        cnarr, segments, variants, show_range, show_gene, window_width
    )
    # Create plots
    if cnarr or segments:
        # Plot CNVs at chromosome level
        if variants:
            # Lay out top 3/5 for the CN scatter, bottom 2/5 for SNP plot
            axgrid = pyplot.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.5)
            axis = pyplot.subplot(axgrid[0])
            axis2 = pyplot.subplot(axgrid[1], sharex=axis)
            # Plot allele freqs for only the selected region
            snv_on_chromosome(
                axis2, sel_snvs, sel_segs, genes, do_trend, by_bin, segment_color
            )
        else:
            _fig, axis = pyplot.subplots()
            if by_bin:
                axis.set_xlabel("Position (bin)")
            else:
                axis.set_xlabel("Position (Mb)")
        axis = cnv_on_chromosome(
            axis,
            sel_probes,
            sel_segs,
            genes,
            antitarget_marker=antitarget_marker,
            do_trend=do_trend,
            x_limits=window_coords,
            y_min=y_min,
            y_max=y_max,
            segment_color=segment_color,
        )
    elif variants:
        # Only plot SNVs in a single-panel layout
        _fig, axis = pyplot.subplots()
        axis = snv_on_chromosome(
            axis, sel_snvs, sel_segs, genes, do_trend, by_bin, segment_color
        )

    if title is None:
        title = "%s %s" % ((cnarr or segments or variants).sample_id, chrom)
    axis.set_title(title, fontsize=14)
    return axis.get_figure()


def select_range_genes(cnarr, segments, variants, show_range, show_gene, window_width):
    """Determine which datapoints to show based on the given options.

    Behaviors::

        start/end   show_gene
           +           +       given region + genes; err if any gene outside it
           -           +       window +/- around genes
           +           -       given region, highlighting any genes within it
           -           -       whole chromosome, no genes

    If `show_range` is a chromosome name only, no start/end positions, then the
    whole chromosome will be shown.

    If region start/end coordinates are given and `show_gene` is '' or ',' (or
    all commas, etc.), then instead of highlighting all genes in the selection,
    no genes will be highlighted.
    """
    chrom, start, end = unpack_range(show_range)
    if start is None and end is None:
        # Either the specified range is only chrom, no start-end, or gene names
        # were given
        window_coords = ()
    else:
        # Viewing region coordinates were specified -- take them as given
        # Fill in open-ended ranges' endpoints
        if start is None:
            start = 0
        elif start < 0:
            start = 0
        if not end:
            # Default selection endpoint to the maximum chromosome position
            end = (cnarr or segments or variants).filter(chromosome=chrom).end.iat[-1]
        if end <= start:
            raise ValueError(
                f"Coordinate range {chrom}:{start}-{end} (from {show_range}) "
                + "has size <= 0"
            )
        window_coords = (start, end)

    gene_ranges = []
    if show_gene is None:
        if window_coords:
            if cnarr:
                # Highlight all genes within the given range
                gene_ranges = plots.gene_coords_by_range(cnarr, chrom, start, end)[
                    chrom
                ]
            if not gene_ranges and (end - start) < 10 * window_width:
                # No genes in the selected region, so if the selection is small
                # (i.e. <80% of the displayed window, <10x window padding),
                # highlight the selected region itself.
                # (To prevent this, use show_gene='' or window_width=0)
                logging.info(
                    "No genes found in selection; will highlight the "
                    "selected region itself instead"
                )
                gene_ranges = [(start, end, "Selection")]
                window_coords = (max(0, start - window_width), end + window_width)

    else:
        gene_names = filter(None, show_gene.split(","))
        if gene_names:
            # Scan for probes matching the specified gene(s)
            gene_coords = plots.gene_coords_by_name(cnarr or segments, gene_names)
            if len(gene_coords) > 1:
                raise ValueError(
                    f"Genes {show_gene} are split across chromosomes "
                    f"{list(gene_coords.keys())}"
                )
            g_chrom, gene_ranges = gene_coords.popitem()
            if chrom:
                # Confirm that the selected chromosomes match
                core.assert_equal(
                    "Chromosome also selected by region (-c) does not match",
                    **{"chromosome": chrom, "gene(s)": g_chrom},
                )
            else:
                chrom = g_chrom

            gene_ranges.sort()
            if window_coords:
                # Verify all genes fit in the given window
                for gene_start, gene_end, gene_name in gene_ranges:
                    if not (start <= gene_start and gene_end <= end):
                        raise ValueError(
                            f"Selected gene {gene_name} "
                            + f"({chrom}:{gene_start}-{gene_end}) "
                            + f"is outside specified region {show_range}"
                        )
            elif not show_range:
                # Set the display window to the selected genes +/- a margin
                window_coords = (
                    max(0, gene_ranges[0][0] - window_width),
                    gene_ranges[-1][1] + window_width,
                )

    # Prune plotted elements to the selected region
    sel_probes = cnarr.in_range(chrom, *window_coords) if cnarr else CNA([])
    sel_segs = (
        segments.in_range(chrom, *window_coords, mode="trim") if segments else CNA([])
    )
    sel_snvs = variants.in_range(chrom, *window_coords) if variants else None
    logging.info(
        "Showing %d probes and %d selected genes in region %s",
        len(sel_probes),
        len(gene_ranges),
        (chrom + ":{}-{}".format(*window_coords) if window_coords else chrom),
    )

    return sel_probes, sel_segs, sel_snvs, window_coords, gene_ranges, chrom


def cnv_on_chromosome(
    axis,
    probes,
    segments,
    genes,
    antitarget_marker=None,
    do_trend=False,
    x_limits=None,
    y_min=None,
    y_max=None,
    segment_color=SEG_COLOR,
):
    """Draw a scatter plot of probe values with optional segments overlaid.

    Parameters
    ----------
    genes : list
        Of tuples: (start, end, gene name)
    """
    # TODO - allow plotting just segments without probes
    # Get scatter plot coordinates
    x = 0.5 * (probes["start"] + probes["end"]) * MB  # bin midpoints
    y = probes["log2"]
    if "weight" in probes:
        w = 46 * probes["weight"] ** 2 + 2
    else:
        w = np.repeat(30, len(x))
    w = w * 1.2
    seg_norm = None
    if segments is not None and len(segments):
        seg_norm = colors.Normalize(vmin=segments.log2.min(), vmax=segments.log2.max())

    # Configure axes
    if not y_min:
        y_min = max(-5.0, min(y.min() - 0.1, -0.3)) if len(y) else -1.1
    if not y_max:
        y_max = max(0.3, y.max() + (0.25 if genes else 0.1)) if len(y) else 1.1
    if x_limits:
        x_min, x_max = x_limits
        axis.set_xlim(x_min * MB, x_max * MB)
    else:
        set_xlim_from(axis, probes, segments)
    setup_chromosome(axis, y_min, y_max, "Copy ratio (log2)")
    if genes:
        highlight_genes(axis, genes, min(2.4, y.max() + 0.1) if len(y) else 0.1)

    if antitarget_marker in (None, "o"):
        # Plot targets and antitargets with the same marker
        axis.scatter(x, y, w, color=POINT_COLOR, alpha=0.4, marker="o")
    else:
        # Use the given marker to plot antitargets separately
        x_fg = []
        y_fg = []
        w_fg = []
        x_bg = []
        y_bg = []
        # w_bg = []
        is_bg = probes["gene"].isin(params.ANTITARGET_ALIASES)
        for x_pt, y_pt, w_pt, is_bg_pt in zip(x, y, w, is_bg):
            if is_bg_pt:
                x_bg.append(x_pt)
                y_bg.append(y_pt)
                # w_bg.append(w_pt)
            else:
                x_fg.append(x_pt)
                y_fg.append(y_pt)
                w_fg.append(w_pt)
        axis.scatter(x_fg, y_fg, w_fg, color=POINT_COLOR, alpha=0.4, marker="o")
        axis.scatter(x_bg, y_bg, color=POINT_COLOR, alpha=0.5, marker=antitarget_marker)

    # Add a local trend line
    if do_trend:
        axis.plot(
            x,
            probes.smooth_log2(),  # .1),
            color=POINT_COLOR,
            linewidth=2,
            zorder=-1,
            snap=False,
        )

    # Draw segments as horizontal lines
    if segments:
        for row in segments:
            color = choose_segment_color(row, norm=seg_norm)
            axis.plot(
                (row.start * MB, row.end * MB),
                (row.log2, row.log2),
                color=color,
                linewidth=4,
                solid_capstyle="round",
                snap=False,
            )
        # Warn about segments masked by default pruning at 'y_min=-5':
        hidden_seg = segments.log2 < y_min
        if hidden_seg.sum():
            logging.warning(
                "WARNING: With 'y_min=%s' %s segments are hidden"
                " --> Add parameter '--y-min %s' to see them",
                y_min,
                hidden_seg.sum(),
                int(np.floor(segments.log2.min())),
            )
            # Signal them as triangles crossing y-axis:
            x_hidden = segments.start[hidden_seg] * MB
            y_hidden = np.array([y_min] * len(x_hidden))
            colors = [_color_from_log2(v, seg_norm) for v in segments.log2[hidden_seg]]
            axis.scatter(
                x_hidden,
                y_hidden,
                marker="^",
                linewidth=3,
                snap=False,
                color=colors,
                edgecolor="none",
                clip_on=False,
                zorder=10,
            )
    if seg_norm is not None:
        sm = cm.ScalarMappable(norm=seg_norm, cmap=SEG_CMAP)
        sm.set_array([])
        axis.get_figure().colorbar(sm, ax=axis, label="Segment log2")
    return axis


def snv_on_chromosome(axis, variants, segments, genes, do_trend, by_bin, segment_color):
    # TODO set x-limits if not already done for probes/segments
    # set_xlim_from(axis, None, segments, variants)
    # setup_chromosome(axis, 0.0, 1.0, "VAF")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("VAF")
    if by_bin:
        axis.set_xlabel("Position (bin)")
    else:
        axis.set_xlabel("Position (Mb)")
    axis.get_yaxis().tick_left()
    axis.get_xaxis().tick_top()
    axis.tick_params(which="both", direction="out", labelbottom=False, labeltop=False)

    x_mb = variants["start"].values * MB
    y = variants["alt_freq"].values
    axis.scatter(x_mb, y, color=POINT_COLOR, alpha=0.3, s=15)
    seg_norm = None
    if segments is not None and len(segments):
        seg_norm = colors.Normalize(vmin=segments.log2.min(), vmax=segments.log2.max())
    if segments or do_trend:
        # Draw average VAF within each segment
        for seg, v_freq in get_segment_vafs(variants, segments):
            if seg:
                posn = [seg.start * MB, seg.end * MB]
                color = choose_segment_color(seg, default_bright=False, norm=seg_norm)
            else:
                posn = [variants.start.iat[0] * MB, variants.start.iat[-1] * MB]
                color = TREND_COLOR
            axis.plot(
                posn,
                [v_freq, v_freq],
                color=color,
                linewidth=2,
                zorder=1,
                solid_capstyle="round",
                snap=False,
            )

    if genes:
        highlight_genes(axis, genes, 0.9)
    return axis


def set_xlim_from(axis, probes=None, segments=None, variants=None):
    """Configure axes for plotting a single chromosome's data.

    Parameters
    ----------
    probes : CopyNumArray
    segments : CopyNumArray
    variants : VariantArray
        All should already be subsetted to the region that will be plotted.
    """
    min_x = np.inf
    max_x = 0
    for arr in (probes, segments, variants):
        if arr and len(arr):
            max_x = max(max_x, arr.end.iat[-1])
            min_x = min(min_x, arr.start.iat[0])
    if max_x <= min_x:
        if min_x != np.inf:
            logging.warning(
                "WARNING: selection start %s > end %s; did you "
                "correctly sort the input file by genomic "
                "location?",
                min_x,
                max_x,
            )
        raise ValueError(
            "No usable data points to plot out of "
            f"{len(probes) if probes else 0} probes, "
            f"{len(segments) if segments else 0} segments, "
            f"{len(variants) if variants else 0} variants"
        )
    axis.set_xlim(min_x * MB, max_x * MB)


def setup_chromosome(axis, y_min=None, y_max=None, y_label=None):
    """Configure axes for plotting a single chromosome's data."""
    if y_min and y_max:
        axis.set_ylim(y_min, y_max)
        if y_min < 0 < y_max:
            axis.axhline(color="k")
    if y_label:
        axis.set_ylabel(y_label)
    axis.tick_params(which="both", direction="out")
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()


# === Shared ===


def choose_segment_color(segment, default_bright=True, norm=None):
    """Choose a display color based on a segment's CNA status.

    Uses the fields added by the 'call' command. If these aren't present, use a
    color derived from :data:`SEG_COLORS`.

    For sex chromosomes, some single-copy deletions or gains might not be
    highlighted, since sample sex isn't used to infer the neutral ploidies.
    """
    highlight_color = _color_from_log2(segment.log2, norm)
    neutral_color = TREND_COLOR
    if "cn" not in segment._fields:
        # No 'call' info
        return highlight_color if default_bright else neutral_color

    # Detect copy number alteration
    expected_ploidies = {"chrY": (0, 1), "Y": (0, 1), "chrX": (1, 2), "X": (1, 2)}
    if segment.cn not in expected_ploidies.get(segment.chromosome, [2]):
        return highlight_color

    # Detect regions of allelic imbalance / LOH
    if (
        segment.chromosome not in expected_ploidies
        and "cn1" in segment._fields
        and "cn2" in segment._fields
        and (segment.cn1 != segment.cn2)
    ):
        return highlight_color

    return neutral_color


def get_segment_vafs(variants, segments):
    """Group SNP allele frequencies by segment.

    Assume variants and segments were already subset to one chromosome.

    Yields
    ------
    tuple
        (segment, value)
    """
    if segments:
        chunks = variants.by_ranges(segments)
    else:
        # Fake segments cover the whole region
        chunks = [(None, variants)]
    for seg, seg_snvs in chunks:
        # ENH: seg_snvs.tumor_boost()
        freqs = seg_snvs["alt_freq"].values
        # Separately emit VAFs above and below .5 for plotting
        idx_above_mid = freqs > 0.5
        for idx_vaf in (idx_above_mid, ~idx_above_mid):
            if sum(idx_vaf) > 1:
                yield (seg, np.median(freqs[idx_vaf]))


def highlight_genes(axis, genes, y_posn):
    """Show gene regions with background color and a text label."""
    # Rotate text in proportion to gene density
    ngenes = len(genes)
    text_size = "small" if ngenes <= 6 else "x-small"
    if ngenes <= 3:
        text_rot = "horizontal"
    elif ngenes <= 6:
        text_rot = 30
    elif ngenes <= 10:
        text_rot = 45
    elif ngenes <= 20:
        text_rot = 60
    else:
        text_rot = "vertical"
    for gene in genes:
        gene_start, gene_end, gene_name = gene
        # Highlight and label gene region
        # (rescale positions from bases to megabases)
        axis.axvspan(
            gene_start * MB, gene_end * MB, alpha=0.5, color=HIGHLIGHT_COLOR, zorder=-1
        )
        axis.text(
            0.5 * (gene_start + gene_end) * MB,
            y_posn,
            gene_name,
            horizontalalignment="center",
            rotation=text_rot,
            size=text_size,
        )
