New Variables
    sixCow          save information of cows
    progress        number of saved information of cows
    interpolate     equation for each unit time
    animStartTime   animation start time
    isMove          if cow is move then isMove is 1 else 0


display()
    using Catmull-Rom, the movement trajectory of the cow per unit time is made into an equation. (unit time : 1 second)

    calculate tangent vector and cow coordinate system


onMouseButton(window, button, state, mods)
    pick six cow.

    if state is GLFW_DOWN then is drag = V_DRAG else if state == GLFW_UP and isDrag != 0 then isDrag = H_DRAG

getPickInfo(window, x, y)
    get cow pick information(bounding box, position, etc)

onMouseDrag(window, x, y)
    if isDrag == V_DRAG and cursorOnCowBoundingBox
    then get ray by screenCoordToRay, calculate the plane normal vector perpendicular to the ray direction and change cow2wld
    else if cursorOnCowBoundingBox
    then get ray by screenCoordToRay, calculate xy-plane and change cow2wld
    