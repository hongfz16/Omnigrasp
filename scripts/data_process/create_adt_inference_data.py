import joblib

if __name__ == '__main__':
    id = "otterSmall"
    sample_data = joblib.load('sample_data/grab_sample.pkl')
    adt_data = joblib.load(f'data/adt/{id}_bps.pkl')

    output_data = {
        id: sample_data["flashlight"],
    }

    output_data[id]['obj_data']['bps_basis'] = adt_data["basis"].detach().cpu()
    output_data[id]['obj_data']['object_code'] = adt_data["bps_code"].detach().cpu()
    output_data[id]['obj_data']['obj_info'][0] = f"phc/data/assets/mesh/adt/{id}.stl"

    joblib.dump(output_data, f'sample_data/{id}_from_flashlight_sample.pkl', compress=True)

