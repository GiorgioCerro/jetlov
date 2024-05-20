# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import click

# import sys
# sys.path.append('/mainfs/home/gc2c20/myproject/hyperparticle/')
# sys.path.append('/mainfs/home/gc2c20/myproject/hyperparticle/data_generation/')
import graphicle as gcl
import numpy as np
from heparchy.write import HdfWriter
from showerpipe.generator import PythiaGenerator
from showerpipe.lhe import LheData, count_events, split
from tqdm import tqdm

print("Everything has been imported")


@click.command()
@click.argument("lhe_path", type=click.Path(exists=True))
@click.argument("pythia_path", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path(path_type=Path))
@click.argument("process_name", type=click.STRING)
def main(lhe_path, pythia_path, output_filepath, process_name):
    """
    Take a lhe file and shower the hard processes with Pythia.
    lhe_path: path to the lhe zipped file
    pythia_path: path to the pythia setting .cmnd
    output_filepath: name and path of the output file
    process_name: it's a tag to the events when written
    """
    # import this module if you want to run on multiple cpus
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    num_procs: int = comm.Get_size()

    total_num_events = count_events(lhe_path)
    print(f"Lhe file contains {total_num_events} events")
    # with stride you split all the events to different processes
    # if you just need a smaller number of event, set the variable to the
    # desired number, e.g. 10 in this case
    stride = 1  # ceil(total_num_events / num_procs)

    # split filepaths for each process
    split_dir = output_filepath.parent
    split_fname = f"{output_filepath.stem}-{rank}{output_filepath.suffix}"
    split_path = split_dir / split_fname

    if rank == 0:  # split up the lhe file
        lhe_splits = split(lhe_path, stride)
        data = next(lhe_splits)
        for i in range(1, num_procs):
            comm.send(next(lhe_splits), dest=i, tag=10 + i)
    else:
        data = comm.recv(source=0, tag=10 + rank)

    # this is the crucial part to reshower
    # .repeat(10) mean that for each event it will repeat the shower 10 times
    # so is stride is 10 and repeat is 10, I will have 100 events,
    # but every batch of 10 events has the same hard process
    repeat = 30
    data = LheData(data).repeat(repeat)
    gen = PythiaGenerator(pythia_path, data)
    gen = tqdm(gen)
    if rank == 0:  # progress bar on root process
        gen = tqdm(gen)
    print("The generator is ready")
    numbers = []
    PT = []
    with HdfWriter(str(split_path)) as hep_file:
        with hep_file.new_process(process_name) as proc:
            shower_id = 0
            total_counts = 0
            counts = 0

            # for ev in tqdm(range(50)):
            #    abc = next(gen)
            # hadronised_gen = repeat_hadronize(gen, repeat)
            for _ in tqdm(range(30000)):
                data = next(lhe_splits)
                if _ < 28000:
                    continue
                data = LheData(data).repeat(repeat)
                gen = PythiaGenerator(pythia_path, data)

                event0 = next(gen)
                graph = gcl.Graphicle.from_event(event0)
                hs_pmu = graph.pmu[graph.hard_mask["intermediate"]]
                if process_name == "background":
                    hs_pmu = graph.pmu[graph.hard_mask["outgoing"]]

                counts = 0
                for idx, event in enumerate(gen):
                    graph = gcl.Graphicle.from_event(event)

                    # this is to cluster the event and find the two jets
                    final = graph.final.data
                    jet_masks = gcl.select.fastjet_clusters(
                        graph.pmu[final], radius=1.0, p_val=-1, top_k=2
                    )
                    # for each jet, if the pt is within range, then we save it
                    for mask in jet_masks:
                        jet_pmu = gcl.MomentumArray(
                            np.sum(graph.pmu[final][mask], axis=0)
                        )
                        if sum(mask) > 10 and jet_pmu.pt >= 500:  # and pt_max <= 550:
                            counts += 1
                            PT.append(jet_pmu.pt)
                            neigh = np.argmin(hs_pmu.delta_R(jet_pmu))
                            with proc.new_event() as event_write:
                                event_write.pmu = graph.pmu.data[final][mask]
                                _shower_id = (shower_id + neigh) * np.ones(sum(mask))
                                event_write.custom["shower_id"] = _shower_id
                                numbers.append(sum(mask))
                        if counts == 10:
                            break
                    if counts >= 10:
                        break
                # print(f"finished shower")
                total_counts += counts
                shower_id += 2
                if total_counts % 1000 == 0:
                    print(f"partial total count: {total_counts}")
            # if (idx + 1) % repeat == 0:
            #    shower_id += 1
            print(f"Generated {total_counts} events")
            mean = np.mean(numbers)
            std = np.std(numbers)
            print(f"number of ptcsl: mean: {mean}- std: {std}")
            print(
                f"""pts: mean: {np.mean(PT)} - std: {np.std(PT)} -
                min: {np.min(PT)} - max: {np.max(PT)}"""
            )

    # stride = 100000  # ceil(total_num_events / num_procs)
    # repeat = 1
    # numbers = []
    # with HdfWriter("valid_" + str(split_path)) as hep_file:
    #    with hep_file.new_process(process_name) as proc:
    #        shower_id = 0
    #        counts = 0
    #        lhe_splits = split(lhe_path, stride)
    #        for _ in range(5):
    #            data = next(lhe_splits)
    #        data = LheData(data).repeat(repeat)
    #        gen = PythiaGenerator(pythia_path, data)
    #        gen = tqdm(gen)
    #        for idx, event in enumerate(gen):
    #            graph = gcl.Graphicle.from_event(event)

    #            # this is to cluster the event and find the two jets
    #            final = graph.final.data
    #            jet_masks = gcl.select.fastjet_clusters(
    #                graph.pmu[final], radius=1., p_val=-1, top_k=2)
    #            # for each jet, if the pt is within range, then we save it
    #            for mask in jet_masks:
    #                jet_pmu = gcl.MomentumArray(np.sum(graph.pmu[final][mask], axis=0))
    #                if sum(mask) > 10 and jet_pmu.pt >= 500:# and pt_max <= 550:
    #                    counts+=1
    #                    with proc.new_event() as event_write:
    #                        event_write.pmu = graph.pmu.data[final][mask]
    #                        #_shower_id = shower_id * np.ones(sum(mask))
    #                        #event_write.custom['shower_id'] = _shower_id
    #                        numbers.append(sum(mask))

    #            if (idx + 1) % repeat == 0:
    #                shower_id += 1
    #            if counts > 100000:
    #                break
    #        print(f"Generated {counts} events")
    #        print(f"""number of ptcsl: mean: {np.mean(numbers)}
    #               - std: {np.std(numbers)}""")

    # stride = 100000
    # repeat = 1
    # numbers = []
    # with HdfWriter("test_" + str(split_path)) as hep_file:
    #    with hep_file.new_process(process_name) as proc:
    #        shower_id = 0
    #        counts = 0
    #        lhe_splits = split(lhe_path, stride)
    #        for _ in range(6):
    #            data = next(lhe_splits)
    #        data = LheData(data).repeat(repeat)
    #        gen = PythiaGenerator(pythia_path, data)
    #        gen = tqdm(gen)
    #        for idx, event in enumerate(gen):
    #            graph = gcl.Graphicle.from_event(event)

    #            # this is to cluster the event and find the two jets
    #            final = graph.final.data
    #            jet_masks = gcl.select.fastjet_clusters(
    #                graph.pmu[final], radius=1.0, p_val=-1, top_k=2
    #            )
    #            # for each jet, if the pt is within range, then we save it
    #            for mask in jet_masks:
    #                jet_pmu = gcl.MomentumArray(np.sum(graph.pmu[final][mask], axis=0))
    #                if sum(mask) > 10 and jet_pmu.pt >= 500:  # and pt_max <= 550:
    #                    counts += 1
    #                    with proc.new_event() as event_write:
    #                        event_write.pmu = graph.pmu.data[final][mask]
    #                        #_shower_id = shower_id * np.ones(sum(mask))
    #                        #event_write.custom["shower_id"] = _shower_id
    #                        numbers.append(sum(mask))

    #            if (idx + 1) % repeat == 0:
    #                shower_id += 1
    #            if counts > 100000:
    #                break
    #        print(f"Generated {counts} events")
    #        print(f"""number of ptcsl: mean: {np.mean(numbers)}
    #                - std: {np.std(numbers)}""")


if __name__ == "__main__":
    sys.exit(main())
